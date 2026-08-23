"""Direct tests for the FV3 duo-cube ModelDriver wiring (slice 1).

Covers the four NEW seams, cheapest-first:

1. ``dcmip16_bc_six_face_state`` — the committed six-face IC assembly
   (window-only fill, zero halos, sphum passenger, NH delz sign).
2. ``FV3DuoDynamicsModel`` — construction refusals fire; one jitted step
   threads the bundle (delp/pt/u/v all move, pytree structure preserved).
3. Component-factory dispatch — the ``fv3_duo`` triple resolves, and the
   slice-1 refusals (physics on, fp32, bad nlev) raise BEFORE any grid
   build.
4. ``ModelDriver._run_fv3_duo`` — the full driver smoke (setup + a short
   run + snapshot files).  This IS the slice-1 driver smoke (a full
   ``run_amip`` invocation adds only argparse on top — covered by the CLI
   round-trip in ``tests/unit/test_run_amip_cli.py``).
5. Restart (slice 2) — ``fv3duo_ckpt_v1`` schema/deck-mismatch refusals,
   checkpoint atomicity, and the PRE-REGISTERED bitwise A/B round-trip
   (straight run vs checkpoint-and-resume) on both the hydro and NH arms.

Cost control: ONE module-scoped grid bundle at N=12/km=5 (the smallest
duo cube the gate files exercise), n_split=2 for the wrapper step (deck
fidelity is the parity runner's job, not this file's), and the driver
smoke runs a fraction of a day.  fp64 + CPU, as the lane requires.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.core.fv3_native_dcmip16_ic import (  # noqa: E402
    dcmip16_bc_six_face_state,
)
from legoesm.core.fv3_native_eta import set_eta_analytic  # noqa: E402
from legoesm.grids.factory import (  # noqa: E402
    FV3DuoGridBundle,
    create_fv3_duo_grid,
)

N, NG, KM = 12, 3, 5
MA = N + 2 * NG
BDT = 120.0


@pytest.fixture(autouse=True)
def _drop_compiled_graphs():
    """Free compiled executables between tests: XLA retains every compiled
    fv_dynamics graph in the process cache, and this file compiles three
    distinct programs (hydro n_split=2, NH n_split=2, driver n_split=8) —
    the retained-graph abort mode test_fv3_dynamics.py measured."""
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def bundle():
    return create_fv3_duo_grid(N, NG)


@pytest.fixture(scope="module")
def eta():
    ak, bk, ptop, _ks = set_eta_analytic(KM)
    return np.asarray(ak), np.asarray(bk), float(ptop)


@pytest.fixture(scope="module")
def model(bundle):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig,
        FV3DuoDynamicsModel,
    )
    # n_split=2 (not the deck's 8): this file gates WIRING, not oracle
    # fidelity — the parity runner owns the deck numbers.
    return FV3DuoDynamicsModel(
        bundle, FV3DuoConfig(km=KM, n_split=2))


# ---------------------------------------------------------------------
# 1. IC assembly
# ---------------------------------------------------------------------

class TestDcmip16SixFaceState:

    def test_hydrostatic_shapes_window_fill_and_zero_halos(self, bundle,
                                                           eta):
        ak, bk, _ptop = eta
        st, sphum = dcmip16_bc_six_face_state(bundle.ctx_np, ak, bk, KM)
        assert len(st) == 6 and len(sphum) == 6
        cs = slice(NG, NG + N)
        for t in range(6):
            assert st[t]["delp"].shape == (MA, MA, KM)
            assert st[t]["u"].shape == (MA, MA + 1, KM)
            assert st[t]["v"].shape == (MA + 1, MA, KM)
            # window filled: analytic delp is strictly positive
            assert (st[t]["delp"][cs, cs, :] > 0.0).all()
            assert (st[t]["pt"][cs, cs, :] > 100.0).all()
            # halos untouched (the oracle leaves them to the in-step
            # exchanges): the first halo ring must be exactly zero
            assert (st[t]["delp"][:NG, :, :] == 0.0).all()
            assert (st[t]["delp"][:, :NG, :] == 0.0).all()
            # sphum: padded, zero halos, non-negative, carries signal
            assert sphum[t].shape == (MA, MA, KM)
            assert (sphum[t][:NG, :, :] == 0.0).all()
            assert sphum[t][cs, cs, :].max() > 0.0
            assert (sphum[t] >= 0.0).all()

    def test_nh_adds_negative_delz_and_zero_w(self, bundle, eta):
        ak, bk, _ptop = eta
        st, _q = dcmip16_bc_six_face_state(bundle.ctx_np, ak, bk, KM,
                                           hydrostatic=False)
        for t in range(6):
            # make_nh: delz strictly negative (z decreases downward in
            # each layer), w identically zero
            assert st[t]["delz"].shape == (N, N, KM)
            assert (st[t]["delz"] < 0.0).all()
            assert (st[t]["w"] == 0.0).all()

    def test_do_pert_changes_the_winds(self, bundle, eta):
        ak, bk, _ptop = eta
        st_p, _ = dcmip16_bc_six_face_state(bundle.ctx_np, ak, bk, KM,
                                            do_pert=True)
        st_0, _ = dcmip16_bc_six_face_state(bundle.ctx_np, ak, bk, KM,
                                            do_pert=False)
        moved = max(float(np.abs(st_p[t]["u"] - st_0[t]["u"]).max())
                    for t in range(6))
        assert moved > 0.0, "do_pert=True did not perturb u (vacuous IC)"


# ---------------------------------------------------------------------
# 2. Grid-factory bundle
# ---------------------------------------------------------------------

class TestGridFactoryEntry:

    def test_bundle_contents(self, bundle):
        assert isinstance(bundle, FV3DuoGridBundle)
        assert bundle.n == N and bundle.ng == NG
        assert bundle.ctx_np["use_ext_bundle"] is True
        # jax ctx carries the attrs fv_dynamics reads
        assert bundle.ctx_jax.n == N and bundle.ctx_jax.ng == NG
        assert bundle.ctx_jax.hs6.shape == (6, MA, MA)

    def test_tiny_resolution_refused(self):
        with pytest.raises(ValueError, match="resolution"):
            create_fv3_duo_grid(2)


# ---------------------------------------------------------------------
# 3. Wrapper: refusals + one threaded step
# ---------------------------------------------------------------------

class TestFV3DuoDynamicsModel:

    def test_bad_km_raises(self, bundle):
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
            FV3DuoConfig,
            FV3DuoDynamicsModel,
        )
        with pytest.raises(ValueError, match="km=7"):
            FV3DuoDynamicsModel(bundle, FV3DuoConfig(km=7))

    def test_positive_kord_tm_raises(self, bundle):
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
            FV3DuoConfig,
            FV3DuoDynamicsModel,
        )
        with pytest.raises(ValueError, match="kord_tm"):
            FV3DuoDynamicsModel(bundle, FV3DuoConfig(kord_tm=9))

    @pytest.mark.parametrize("kords", [
        dict(kord_mt=13),
        dict(kord_tr=7),
        dict(kord_tm=-7),
    ])
    def test_non_pinned_kord_deck_refused_at_construction(self, bundle,
                                                          kords):
        """Certification is DECK-PINNED to (9, -9, 9): any other order is
        refused at construction, not at the first remap on step 1."""
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
            FV3DuoConfig,
            FV3DuoDynamicsModel,
        )
        with pytest.raises(ValueError, match="DECK-PINNED"):
            FV3DuoDynamicsModel(bundle, FV3DuoConfig(km=KM, **kords))

    def test_wrong_grid_type_raises(self):
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
            FV3DuoDynamicsModel,
        )
        with pytest.raises(TypeError, match="FV3DuoGridBundle"):
            FV3DuoDynamicsModel(object())

    def test_step_rejects_a_non_bundle_state(self, model):
        with pytest.raises(ValueError, match="keys"):
            model.step({"delp": jnp.zeros(3)}, BDT)

    def test_one_step_threads_the_bundle(self, model):
        """One jitted step: every prognostic moves, structure preserved."""
        ic = model.dcmip16_initial_state()
        assert set(ic) == {"state", "press", "q", "omga", "nh"}
        assert ic["nh"] is None  # hydrostatic arm
        out = model.step(ic, BDT)
        assert set(out) == set(ic)
        assert set(out["state"]) == set(ic["state"])
        assert set(out["press"]) == set(ic["press"])
        assert len(out["q"]) == 1
        for nm in ("delp", "pt", "u", "v"):
            a, b = np.asarray(ic["state"][nm]), np.asarray(out["state"][nm])
            assert a.shape == b.shape
            assert np.isfinite(b).all(), f"{nm} went non-finite in one step"
            assert float(np.abs(b - a).max()) > 0.0, f"{nm} did not move"
        # pt stays TEMPERATURE (the theta_v round trip closed): range gate
        cs = slice(NG, NG + N)
        pt_win = np.asarray(out["state"]["pt"])[:, cs, cs, :]
        assert 150.0 < pt_win.min() and pt_win.max() < 400.0, (
            f"pt window [{pt_win.min():.1f}, {pt_win.max():.1f}] K is not "
            f"a temperature — the theta_v conversion did not round-trip")
        # sphum stays a passenger: total tracer mass moves only by
        # transport (finite + bounded), never NaN
        q1 = np.asarray(out["q"][0])
        assert np.isfinite(q1).all()

    def test_validate_dycore_contract(self, model):
        from legoesm.components.protocol import validate_dycore
        validate_dycore(model)

    def test_nh_arm_one_step(self, bundle):
        """The NH arm (model_type='nonhydrostatic'): carry prebuilt, one
        step moves delp/pt/u/v AND w/delz, structure preserved."""
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
            FV3DuoConfig,
            FV3DuoDynamicsModel,
        )
        m = FV3DuoDynamicsModel(
            bundle, FV3DuoConfig(km=KM, n_split=2, hydrostatic=False))
        ic = m.dcmip16_initial_state()
        assert ic["nh"] is not None, "NH carry was not prebuilt"
        out = m.step(ic, BDT)
        assert set(out) == set(ic)
        for nm in ("delp", "pt", "u", "v", "w", "delz"):
            a, b = np.asarray(ic["state"][nm]), np.asarray(out["state"][nm])
            assert a.shape == b.shape
            assert np.isfinite(b).all(), f"NH {nm} went non-finite"
        # delz must have moved (it is prognostic on the NH arm); w starts
        # at 0 and picks up a small but nonzero signal in one step
        assert float(np.abs(np.asarray(out["state"]["delz"])
                            - np.asarray(ic["state"]["delz"])).max()) > 0.0
        assert float(np.abs(np.asarray(out["state"]["w"])).max()) > 0.0


# ---------------------------------------------------------------------
# 4. Component-factory dispatch + refusals
# ---------------------------------------------------------------------

def _fv3_duo_config(**over):
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    grid = GridConfig(grid_type="cubed_sphere", resolution=N, nlev=KM)
    dycore = DycoreConfig(model_type=over.pop("model_type", "hydrostatic"),
                          discretization="fv3_duo",
                          dt=over.pop("dt", 1920.0))
    base = dict(
        grid=grid, dycore=dycore, days=1,
        radiation="none", convection="none", microphysics="none",
        turbulence="none", gravity_wave_drag="none",
        precision="fp64",
        output=OutputConfig(diag_days=1,
                            checkpoint_days=over.pop("checkpoint_days", 0),
                            output_dir=over.pop("output_dir", "")),
    )
    base.update(over)
    return ExperimentConfig(**base)


class TestComponentFactoryDispatch:

    def test_triples_registered(self):
        from legoesm.driver.component_factory import _DRIVER_SUPPORTED
        assert _DRIVER_SUPPORTED[
            ("hydrostatic", "fv3_duo", "cubed_sphere")
        ] == "fv3_duo_primitive_equations"
        assert _DRIVER_SUPPORTED[
            ("nonhydrostatic", "fv3_duo", "cubed_sphere")
        ] == "fv3_duo_primitive_equations"

    @pytest.mark.parametrize("bad, frag", [
        (dict(radiation="gray"), "silently inert"),
        (dict(convection="sbm"), "silently inert"),
        (dict(microphysics="sundqvist"), "silently inert"),
        (dict(turbulence="louis"), "silently inert"),
        (dict(gravity_wave_drag="rayleigh"), "silently inert"),
        (dict(precision="fp32"), "fp64"),
        (dict(held_suarez_forcing=True), "Held-Suarez"),
        (dict(distributed=True), "single-process"),
    ])
    def test_slice1_refusals_fire(self, bad, frag):
        """Every refusal raises BEFORE any (expensive) duo grid build,
        with the refusal-SPECIFIC message fragment (a loose "fv3_duo"
        match would pass on any unrelated error on the same path)."""
        from legoesm.driver.component_factory import create_atmosphere_dycore
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        cfg = _fv3_duo_config(**bad)
        with pytest.raises(ValueError, match=frag):
            create_atmosphere_dycore(cfg, create_cubed_sphere(N),
                                     create_sigma_coordinate(KM))

    def test_bad_nlev_refused(self):
        from legoesm.driver.component_factory import create_atmosphere_dycore
        from legoesm.driver.config import GridConfig
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        cfg = _fv3_duo_config()
        cfg = cfg._replace(grid=GridConfig(grid_type="cubed_sphere",
                                           resolution=N, nlev=7))
        with pytest.raises(ValueError, match="nlev"):
            create_atmosphere_dycore(cfg, create_cubed_sphere(N),
                                     create_sigma_coordinate(7))

    def test_solver_name_resolves_and_matrix_is_synced(self):
        from legoesm.atmosphere.dynamics import (
            get_solver_class,
            resolve_solver_name,
        )
        from legoesm.supported_matrix import canonical_solver_names
        assert resolve_solver_name(
            dynamics="hydrostatic", discretization="fv3_duo",
        ) == "fv3_duo_primitive_equations"
        assert resolve_solver_name(
            dynamics="nonhydrostatic", discretization="fv3_duo",
        ) == "fv3_duo_primitive_equations"
        cls = get_solver_class("fv3_duo_primitive_equations")
        assert cls.__name__ == "FV3DuoDynamicsModel"
        assert "fv3_duo_primitive_equations" in canonical_solver_names(
            "atmosphere")


class TestFV3DuoDefaultDenyWall:
    """The refusal wall is DEFAULT-DENY (codex 2026-08-18 BLOCKERs): any
    field differing from its ExperimentConfig default and not on the
    explicit allow list is refused, all offenders listed in ONE error."""

    # One representative smuggler per BLOCKER class: a moisture flag, a
    # land field, a forcing-deck path.  Each reached _run_fv3_duo
    # silently under the enumerated deny-list.
    SMUGGLERS = [
        ("hard_saturation_adjustment", True),
        ("use_multilayer_land", True),
        ("forcing_path", "/data/era5_sst.nc"),
    ]

    def test_default_plus_allowed_config_passes(self):
        from legoesm.driver.component_factory import (
            _refuse_fv3_duo_non_default,
        )
        _refuse_fv3_duo_non_default(_fv3_duo_config())  # must not raise
        # slice-2 restart: the shared checkpoint cadence is allow-listed
        _refuse_fv3_duo_non_default(_fv3_duo_config(checkpoint_days=1))

    @pytest.mark.parametrize("field, value", SMUGGLERS)
    def test_each_smuggler_refused_with_path_named(self, field, value):
        from legoesm.driver.component_factory import (
            _refuse_fv3_duo_non_default,
        )
        with pytest.raises(ValueError, match=field):
            _refuse_fv3_duo_non_default(_fv3_duo_config(**{field: value}))

    def test_all_offenders_listed_at_once(self):
        from legoesm.driver.component_factory import (
            _refuse_fv3_duo_non_default,
        )
        cfg = _fv3_duo_config(**dict(self.SMUGGLERS))
        with pytest.raises(ValueError) as ei:
            _refuse_fv3_duo_non_default(cfg)
        msg = str(ei.value)
        for field, _v in self.SMUGGLERS:
            assert field in msg, f"error message omits offender {field}"

    def test_nested_offender_named_by_dotted_path(self):
        from legoesm.driver.component_factory import (
            _refuse_fv3_duo_non_default,
        )
        cfg0 = _fv3_duo_config()
        cfg = cfg0._replace(
            output=cfg0.output._replace(checkpoint_format="zarr"))
        with pytest.raises(ValueError, match="output.checkpoint_format"):
            _refuse_fv3_duo_non_default(cfg)

    def test_wall_is_wired_into_the_factory_branch(self):
        from legoesm.driver.component_factory import create_atmosphere_dycore
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        cfg = _fv3_duo_config(use_multilayer_land=True)
        with pytest.raises(ValueError, match="use_multilayer_land"):
            create_atmosphere_dycore(cfg, create_cubed_sphere(N),
                                     create_sigma_coordinate(KM))

    def test_matrix_declares_both_dynamics_axes(self):
        """NH support is STRUCTURAL, not a note (codex 2026-08-18).

        The first version of this asserted a free-text ``note`` said
        "nonhydrostatic" -- but the matrix's only production consumer
        ignores notes, so a caller filtering on ``dynamics`` still
        concluded NH was unsupported. There is now one ROW per axis
        (uniqueness re-keyed to the (canonical_name, dynamics) pair),
        and this asserts what a consumer can actually read.
        """
        from legoesm.supported_matrix import ATMOSPHERE_MATRIX
        entries = [e for e in ATMOSPHERE_MATRIX
                   if e.canonical_name == "fv3_duo_primitive_equations"]
        assert {e.dynamics for e in entries} == {"hydrostatic",
                                                 "nonhydrostatic"}, \
            [e.dynamics for e in entries]
        assert len({e.class_name for e in entries}) == 1, (
            "both axes must resolve to the ONE certified class")


# ---------------------------------------------------------------------
# 5. ModelDriver smoke — the slice-1 driver lane, end to end
# ---------------------------------------------------------------------

class TestModelDriverLane:

    def test_short_run_completes_and_writes_snapshots(self, tmp_path):
        """Full setup + a few duo steps + snapshot files on disk.

        ``days=1`` at dt=1920 s = 45 steps at C12/km=5 — a couple of
        minutes on CPU, dominated by the one-time jit compile.  This is
        the slice-1 driver smoke (run_amip adds only argparse on top).
        """
        from legoesm.driver.model_driver import ModelDriver
        cfg = _fv3_duo_config(output_dir=str(tmp_path))
        driver = ModelDriver(cfg, output_dir=tmp_path)
        driver.setup()
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
            FV3DuoDynamicsModel,
        )
        assert isinstance(driver.model, FV3DuoDynamicsModel)
        status = driver.run()
        assert status == "COMPLETED", f"driver lane returned {status!r}"
        snaps = sorted(tmp_path.glob("fv3duo_snapshot_step_*.npz"))
        assert snaps, "no duo snapshots were written"
        # _create_dycore may CFL-clamp dt (1920 s exceeds the C12
        # primitive-eq envelope), so compute the expected step count from
        # the POST-setup dt, not the requested one.
        n_expect = int(cfg.days * 86400.0 / driver.config.dycore.dt)
        with np.load(snaps[-1]) as f:
            assert int(f["_step"]) == n_expect
            for nm in ("delp", "pt", "u", "v", "ps", "q0"):
                assert nm in f.files, f"snapshot missing {nm}"
                assert np.isfinite(f[nm]).all()
        # final-state digest source: the driver handed the ACTUAL bundle
        assert isinstance(driver.state, dict)
        assert set(driver.state) == {"state", "press", "q", "omga", "nh"}
        # Physical invariants (codex MINOR: finiteness alone passes a
        # physically corrupt state).  (1) Global dry mass — the certified
        # flux-form core conserves area-weighted sum(delp) to fp64
        # roundoff; compare the deterministic analytic IC to the final
        # bundle over the compute windows.
        ic = driver.model.dcmip16_initial_state(do_pert=True)
        ng_, n_ = driver.model.grid.ng, driver.model.grid.n
        cs = slice(ng_, ng_ + n_)
        areas = np.stack([
            np.asarray(driver.model.grid.ctx_np["gs6"][t]["area"])[cs, cs]
            for t in range(6)])
        assert (areas > 0.0).all()  # windows never see the halo poison

        def _dry_mass(bundle):
            delp = np.asarray(bundle["state"]["delp"])[:, cs, cs, :]
            return float((delp.sum(axis=-1) * areas).sum())

        m0, m1 = _dry_mass(ic), _dry_mass(driver.state)
        drift = abs(m1 - m0) / m0
        # Bound scope (codex 2026-08-18): 1e-12 is supported by ONE
        # measurement — rel = 0.0 at THIS config (C12/km=5, hydro,
        # this step count). It does not establish scaling with
        # resolution, km, NH, or duration; a new config re-measures.
        print(f"FV3DUO_DRY_MASS_DRIFT rel={drift:.3e} over {n_expect} steps")
        assert drift < 1e-12, (
            f"global dry-mass drift {drift:.3e} over {n_expect} steps "
            f"exceeds the fp64-roundoff envelope 1e-12")
        # (2) The driver's own 400 m/s wind envelope holds at the end.
        umax = max(
            float(np.abs(np.asarray(driver.state["state"][nm])).max())
            for nm in ("u", "v"))
        assert umax < 400.0, f"final max|wind|={umax:.1f} m/s >= 400"
        # (3) Explicit terminal-status marker next to the snapshots.
        assert (tmp_path / "fv3duo_status.txt").read_text().strip() \
            == "COMPLETED"

    def test_blowup_writes_explicit_status_marker(self, tmp_path):
        """A guard-tripped run leaves an EXPLICIT marker (not just a
        missing manifest digest).  Cheap: an identity step (no jit
        compile) plus a 0 m/s envelope trips the guard at the first
        snapshot."""
        from legoesm.driver.model_driver import ModelDriver
        cfg = _fv3_duo_config(output_dir=str(tmp_path))
        driver = ModelDriver(cfg, output_dir=tmp_path)
        driver.setup()
        driver.model.step = lambda bundle, dt: bundle
        driver._FV3_DUO_BLOWUP_UMAX_MS = 0.0
        status = driver.run()
        assert status.startswith("BLOWUP"), f"got {status!r}"
        marker = (tmp_path / "fv3duo_status.txt").read_text().strip()
        assert marker == status


# ---------------------------------------------------------------------
# 6. Restart (slice 2): fv3duo_ckpt_v1 refusals + the bitwise A/B gate
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def restart_driver(tmp_path_factory):
    """ONE set-up duo driver shared by the (cheap, no-stepping) refusal
    tests — setup builds the grid + model but compiles nothing."""
    from legoesm.driver.model_driver import ModelDriver
    d = tmp_path_factory.mktemp("fv3duo_restart_refusals")
    cfg = _fv3_duo_config(output_dir=str(d))
    drv = ModelDriver(cfg, output_dir=d)
    drv.setup()
    return drv, d


def _write_duo_ckpt(path, drv, **over):
    """Hand-built npz whose metadata is VALID for *drv*; each refusal
    test overrides ONE field (the metadata gates fire before any array
    decode, so refusal tests need no arrays)."""
    meta = {
        "_schema": "fv3duo_ckpt_v1",
        "_step": np.int64(3),
        "_day": np.float64(0.25),
        "_dt": np.float64(drv.config.dycore.dt),
        "_hydrostatic": np.bool_(drv.model.config.hydrostatic),
        "_km": np.int64(drv.model.config.km),
        "_resolution": np.int64(drv.model.grid.n),
        "_git_sha": "test",
    }
    meta.update(over)
    with open(path, "wb") as fh:
        np.savez(fh, **meta)
    return path


class TestFV3DuoRestart:

    def test_load_before_setup_refused(self, tmp_path):
        """The loader validates km/resolution/hydrostatic against the
        CONSTRUCTED model, so a pre-setup load must refuse loudly."""
        from legoesm.driver.model_driver import ModelDriver
        cfg = _fv3_duo_config(output_dir=str(tmp_path))
        driver = ModelDriver(cfg, output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="setup"):
            driver.load_checkpoint(tmp_path / "x.npz")

    def test_missing_file_refused(self, restart_driver):
        drv, d = restart_driver
        with pytest.raises(FileNotFoundError, match="not found"):
            drv.load_checkpoint(d / "no_such_checkpoint.npz")

    def test_foreign_schema_refused_before_any_decode(self, restart_driver):
        """A cube-style npz (no _schema) is refused by NAME of the duo
        schema — never half-decoded into a shape error."""
        drv, d = restart_driver
        p = d / "foreign.npz"
        np.savez(p, u=np.zeros(3), step=np.int64(1))
        with pytest.raises(ValueError, match="fv3duo_ckpt_v1"):
            drv.load_checkpoint(p)

    @pytest.mark.parametrize("field, value, frag", [
        ("_km", 10, "km mismatch"),
        ("_resolution", 24, "resolution mismatch"),
        ("_hydrostatic", False, "hydrostatic mismatch"),
        ("_dt", 7.0, "dt mismatch"),
    ])
    def test_deck_mismatch_refused(self, restart_driver, field, value,
                                   frag):
        drv, d = restart_driver
        p = _write_duo_ckpt(d / f"mm{field}.npz", drv, **{field: value})
        with pytest.raises(ValueError, match=frag):
            drv.load_checkpoint(p)

    def test_truncated_checkpoint_refused(self, restart_driver):
        """Valid metadata but no arrays -> named truncation refusal."""
        drv, d = restart_driver
        p = _write_duo_ckpt(d / "truncated.npz", drv)
        with pytest.raises(ValueError, match="truncated"):
            drv.load_checkpoint(p)

    def test_lossy_leaf_refused(self, restart_driver):
        """An f32 array under the fp64 schema is a leaf that lost bits
        — refused by name, never silently promoted."""
        drv, d = restart_driver
        p = d / "lossy.npz"
        meta = {
            "_schema": "fv3duo_ckpt_v1", "_step": np.int64(3),
            "_day": np.float64(0.25),
            "_dt": np.float64(drv.config.dycore.dt),
            "_hydrostatic": np.bool_(True), "_km": np.int64(KM),
            "_resolution": np.int64(N), "_git_sha": "test",
        }
        with open(p, "wb") as fh:
            np.savez(fh, state_u=np.zeros(2, dtype=np.float32), **meta)
        with pytest.raises(ValueError, match="float64"):
            drv.load_checkpoint(p)

    def test_restart_chain_is_bitwise(self, tmp_path):
        """TWO bounces, not one (GLM 2026-08-19).

        A single split point cannot see a bug phase-locked to the
        cadence or to the first-restarted step: N -> ckpt -> M -> ckpt
        -> K must equal N+M+K straight through. Also exercises loading a
        checkpoint that was itself written by a restarted run.
        """
        from legoesm.driver.model_driver import ModelDriver
        mk = dict(days=3, checkpoint_days=1, model_type="hydrostatic")
        da, db = tmp_path / "ca", tmp_path / "cb"
        da.mkdir(), db.mkdir()
        drv_a = ModelDriver(_fv3_duo_config(output_dir=str(da), **mk),
                            output_dir=da)
        drv_a.setup()
        assert drv_a.run() == "COMPLETED"
        dt = drv_a.config.dycore.dt
        n1 = max(1, int(86400.0 / dt))

        # bounce 1: fresh driver from day 1
        drv_b = ModelDriver(_fv3_duo_config(output_dir=str(db), **mk),
                            output_dir=db)
        drv_b.setup()
        st, dy = drv_b.load_checkpoint(
            da / f"fv3duo_ckpt_step_{n1:09d}.npz")
        assert drv_b.run(start_step=st, start_day=dy) == "COMPLETED"
        # bounce 2: another fresh driver, from the checkpoint B WROTE at
        # day 2 -- i.e. a restart of a restart.
        dc = tmp_path / "cc"
        dc.mkdir()
        drv_c = ModelDriver(_fv3_duo_config(output_dir=str(dc), **mk),
                            output_dir=dc)
        drv_c.setup()
        st2, dy2 = drv_c.load_checkpoint(
            db / f"fv3duo_ckpt_step_{2 * n1:09d}.npz")
        assert drv_c.run(start_step=st2, start_day=dy2) == "COMPLETED"

        def _walk(b):
            out = {}
            for k, v in b["state"].items():
                out[f"state.{k}"] = np.asarray(v)
            for k, v in b["press"].items():
                out[f"press.{k}"] = np.asarray(v)
            for i, qt in enumerate(b["q"]):
                out[f"q[{i}]"] = np.asarray(qt)
            out["omga"] = np.asarray(b["omga"])
            return out

        fa, fc = _walk(drv_a.state), _walk(drv_c.state)
        assert set(fa) == set(fc)
        diffs = [nm for nm in sorted(fa)
                 if fa[nm].shape != fc[nm].shape
                 or fa[nm].dtype != fc[nm].dtype
                 or fa[nm].tobytes() != fc[nm].tobytes()]
        assert not diffs, f"restart CHAIN is not bitwise: {diffs}"

    def test_start_day_must_match_the_checkpoint(self, tmp_path):
        """The (step, day) pair is the checkpoint's, not the caller's.

        codex MAJOR: only start_step was bound, so a caller could pass
        any start_day and the run wrote snapshots stamped with a shifted
        day. The dry dynamics never reads day, so the bitwise state test
        stayed green while provenance drifted.
        """
        from legoesm.driver.model_driver import ModelDriver
        d = tmp_path / "sd"
        d.mkdir()
        cfg = _fv3_duo_config(output_dir=str(d), days=1, checkpoint_days=1)
        drv = ModelDriver(cfg, output_dir=d)
        drv.setup()
        assert drv.run() == "COMPLETED"
        ck = sorted(d.glob("fv3duo_ckpt_step_*.npz"))[-1]
        # FRESH directory for the second config: the run manifest refuses
        # to mix two configs' provenance in one directory, and days=1 ->
        # days=2 is a different config (that guard is right; my first
        # version of this test tripped it).
        d2 = tmp_path / "sd2"
        d2.mkdir()
        drv2 = ModelDriver(_fv3_duo_config(output_dir=str(d2), days=2,
                                           checkpoint_days=1),
                           output_dir=d2)
        drv2.setup()
        step, day = drv2.load_checkpoint(ck)
        with pytest.raises(ValueError, match="does not match the loaded"):
            drv2.run(start_step=step, start_day=day + 0.5)

    def test_bare_start_step_refused(self, restart_driver):
        """A start_step without a load_checkpoint-staged bundle has no
        state to resume from — refused, pointing at load_checkpoint."""
        drv, _d = restart_driver
        with pytest.raises(ValueError, match="load_checkpoint"):
            drv._run_fv3_duo(start_step=7)

    @pytest.mark.parametrize("model_type", ["hydrostatic",
                                            "nonhydrostatic"])
    def test_restart_roundtrip_bitwise(self, tmp_path, model_type):
        """PRE-REGISTERED acceptance (non-negotiable): run A = 2 days
        straight; run B = fresh driver loading A's day-1 checkpoint,
        then the remaining day.  Final bundles must be BITWISE identical
        on EVERY array (same jitted program; fp64 npz round-trip is
        exact).  A tolerance here would hide state loss — if this
        fails, that is a FINDING, not a bound to relax."""
        from legoesm.driver.model_driver import ModelDriver
        dir_a, dir_b = tmp_path / "a", tmp_path / "b"
        dir_a.mkdir(), dir_b.mkdir()
        mk = dict(days=2, checkpoint_days=1, model_type=model_type)
        cfg_a = _fv3_duo_config(output_dir=str(dir_a), **mk)
        drv_a = ModelDriver(cfg_a, output_dir=dir_a)
        drv_a.setup()
        assert drv_a.run() == "COMPLETED"
        dt = drv_a.config.dycore.dt  # post-CFL-clamp effective dt
        n_total = int(2 * 86400.0 / dt)
        n_mid = max(1, int(1 * 86400.0 / dt))
        mid = dir_a / f"fv3duo_ckpt_step_{n_mid:09d}.npz"
        final_a = dir_a / f"fv3duo_ckpt_step_{n_total:09d}.npz"
        assert mid.is_file() and final_a.is_file()
        # atomicity: a successful run never leaves the tmp sibling
        assert not list(dir_a.glob("*.tmp")), "atomic write leaked .tmp"

        cfg_b = _fv3_duo_config(output_dir=str(dir_b), **mk)
        drv_b = ModelDriver(cfg_b, output_dir=dir_b)
        drv_b.setup()
        step, day = drv_b.load_checkpoint(mid)
        assert step == n_mid
        assert day == pytest.approx(n_mid * dt / 86400.0)
        assert drv_b.run(start_step=step, start_day=day) == "COMPLETED"
        assert not list(dir_b.glob("*.tmp"))

        # INDEPENDENT ENUMERATION (codex MAJOR 2026-08-19): comparing via
        # the writer's own _fv3_duo_flatten_bundle made the "EVERY array"
        # claim circular -- deleting a member from that helper would drop
        # it from persistence AND from this comparison at once, and q and
        # omga are exactly the members that would not otherwise show
        # (q is dynamically passive at zvir=0; omga is output-only).
        # This walks the bundle structure directly instead.
        def _walk(b):
            out = {}
            for k, v in b["state"].items():
                out[f"state.{k}"] = np.asarray(v)
            for k, v in b["press"].items():
                out[f"press.{k}"] = np.asarray(v)
            for i, qt in enumerate(b["q"]):
                out[f"q[{i}]"] = np.asarray(qt)
            out["omga"] = np.asarray(b["omga"])
            if b.get("nh") is not None:
                for k, v in b["nh"].items():
                    out[f"nh.{k}"] = np.asarray(v)
            return out

        fa, fb = _walk(drv_a.state), _walk(drv_b.state)
        assert set(fa) == set(fb), (sorted(set(fa) ^ set(fb)))
        # the enumeration must actually cover the contract
        assert any(k.startswith("q[") for k in fa), "no tracer compared"
        assert "omga" in fa
        if model_type == "nonhydrostatic":
            assert any(k.startswith("nh.") for k in fa), \
                "NH arm persisted no nh carry"
        # dtype BEFORE tobytes (GLM MINOR): equal byte counts compare
        # equal across float64x3 vs int64x3, so bytes alone leaves a hole.
        diffs = [nm for nm in sorted(fa)
                 if fa[nm].shape != fb[nm].shape
                 or fa[nm].dtype != fb[nm].dtype
                 or fa[nm].tobytes() != fb[nm].tobytes()]
        assert not diffs, (
            f"restart is NOT bitwise; differing arrays: {diffs}")

        if model_type == "hydrostatic":
            # total-days contract: restarting from the FINAL checkpoint
            # with the same --days is at/past target -> loud refusal.
            cfg_c = _fv3_duo_config(output_dir=str(dir_b), **mk)
            drv_c = ModelDriver(cfg_c, output_dir=dir_b)
            drv_c.setup()
            step_c, day_c = drv_c.load_checkpoint(final_a)
            assert step_c == n_total
            with pytest.raises(ValueError, match="nothing to run"):
                drv_c.run(start_step=step_c, start_day=day_c)


# =====================================================================
# The default-deny wall's own contract (codex BLOCKER + GLM MAJORs,
# 2026-08-18): the wall diffs against ExperimentConfig() AT CALL TIME,
# so a future default change would silently widen it. These tests
# convert that drift into a review-gated CI failure.
# =====================================================================

def _wall_surface():
    from legoesm.driver.component_factory import _flatten_config_fields
    from legoesm.driver.config import ExperimentConfig
    return sorted(_flatten_config_fields(ExperimentConfig()))


def test_wall_default_surface_is_frozen():
    """Golden hash of the flattened default surface.

    Changing ANY ExperimentConfig default (or the flattener's shape)
    moves this hash; updating the literal below is the review gate at
    which _FV3_DUO_ALLOWED_NONDEFAULT must be re-justified — without
    this, a default flipping to a value the duo lane silently ignores
    recreates the "successful wrong experiment" (codex BLOCKER).
    """
    import hashlib
    digest = hashlib.sha256(repr(_wall_surface()).encode()).hexdigest()
    assert digest == _WALL_SURFACE_SHA256, (
        f"ExperimentConfig's flattened default surface changed "
        f"(sha256 {digest}). Re-review _FV3_DUO_ALLOWED_NONDEFAULT in "
        f"component_factory.py against the new defaults, then update "
        f"_WALL_SURFACE_SHA256 in this test.")


# ponytail: filled by the first CI run's failure message; the VALUE is
# the reviewable artifact, the mechanism is above.
_WALL_SURFACE_SHA256 = "7fb1bc3750cdf11d0ba437879c131a1b1edb8db5fa70550b9bec0caece2efb49"


def test_wall_leaf_types_are_scalar():
    """Every flattened leaf is a scalar/tuple — no dict/list/ndarray.

    GLM 2026-08-18: a mutable leaf (shared class-level dict mutated in
    place) compares EQUAL to the default it aliases and passes the wall
    silently; an ndarray leaf makes `==` ambiguous and crashes it. This
    census makes any future such field fail here, in CI, with a named
    path — before it can reach the wall at runtime.
    """
    import enum
    ok = (str, int, float, bool, type(None), tuple, enum.Enum)
    bad = [(p, type(v).__name__) for p, v in _wall_surface()
           if not isinstance(v, ok)]
    assert not bad, f"non-scalar config leaves (wall hazard): {bad}"


def test_wall_allowlist_paths_are_live():
    """Every allow-listed path exists in the surface (GLM: a stale entry
    after a field rename is dead weight AND the rename escaped
    re-justification)."""
    from legoesm.driver.component_factory import _FV3_DUO_ALLOWED_NONDEFAULT
    paths = {p for p, _ in _wall_surface()}
    dead = _FV3_DUO_ALLOWED_NONDEFAULT - paths
    assert not dead, f"allow-listed paths not in the config surface: {dead}"


def test_the_driver_can_only_build_the_certified_split_counts():
    """The remap orders are refused at construction if they leave the deck;
    the acoustic split counts are not, so a reviewer read them as an escape.

    They are not one, because nothing plumbs them: the factory builds the
    solver's configuration itself and passes only the level count and the
    hydrostatic switch, so a launch cannot ask for a different number of
    acoustic substeps per remap. This test is that closure, pinned — it goes
    red the day someone threads a split count through the experiment
    configuration without also certifying it (codex).
    """
    import ast
    import inspect

    from legoesm.driver import component_factory

    src = inspect.getsource(component_factory.create_atmosphere_dycore)
    tree = ast.parse(src.lstrip() if src[0] not in " \t" else
                     __import__("textwrap").dedent(src))
    built = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "FV3DuoConfig"]
    assert built, "the duo lane no longer builds its own FV3DuoConfig here"
    for call in built:
        passed = {kw.arg for kw in call.keywords}
        assert not ({"k_split", "n_split"} & passed), (
            f"the factory now passes {sorted({'k_split', 'n_split'} & passed)} "
            "into the duo solver, so a launch can select an acoustic split "
            "count the oracle parity never measured; refuse non-certified "
            "values at construction the way the remap orders are refused")

    # And the defaults it therefore gets are the ones the parity was run at.
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import FV3DuoConfig
    assert (FV3DuoConfig().k_split, FV3DuoConfig().n_split) == (1, 8)
