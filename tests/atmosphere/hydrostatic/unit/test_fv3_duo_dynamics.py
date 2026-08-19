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
        cfg = cfg0._replace(output=cfg0.output._replace(checkpoint_days=5))
        with pytest.raises(ValueError, match="output.checkpoint_days"):
            _refuse_fv3_duo_non_default(cfg)

    def test_wall_is_wired_into_the_factory_branch(self):
        from legoesm.driver.component_factory import create_atmosphere_dycore
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        cfg = _fv3_duo_config(use_multilayer_land=True)
        with pytest.raises(ValueError, match="use_multilayer_land"):
            create_atmosphere_dycore(cfg, create_cubed_sphere(N),
                                     create_sigma_coordinate(KM))

    def test_matrix_note_declares_nh_support(self):
        from legoesm.supported_matrix import ATMOSPHERE_MATRIX
        entries = [e for e in ATMOSPHERE_MATRIX
                   if e.canonical_name == "fv3_duo_primitive_equations"]
        # TWO rows, one per dynamics axis, because one class carries both
        # behind a static switch. The test asserted a single row and then
        # read its note, so it was reading the hydrostatic row -- which has
        # none -- and had never passed.
        assert {e.dynamics for e in entries} == {"hydrostatic",
                                                 "nonhydrostatic"}
        nh = [e for e in entries if e.dynamics == "nonhydrostatic"]
        assert len(nh) == 1 and "hydrostatic switch" in nh[0].note


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

    def test_restart_refused_before_any_decode(self, tmp_path):
        """run_amip calls load_checkpoint BEFORE the lane's own check —
        the refusal must fire before any decode or existence check
        (a nonexistent path must hit the refusal, not FileNotFoundError)."""
        from legoesm.driver.model_driver import ModelDriver
        cfg = _fv3_duo_config(output_dir=str(tmp_path))
        driver = ModelDriver(cfg, output_dir=tmp_path)
        with pytest.raises(NotImplementedError, match="decode"):
            driver.load_checkpoint(tmp_path / "no_such_checkpoint.npz")

    def test_restart_refused(self, tmp_path):
        from legoesm.driver.model_driver import ModelDriver
        cfg = _fv3_duo_config(output_dir=str(tmp_path))
        driver = ModelDriver(cfg, output_dir=tmp_path)
        driver.setup()
        with pytest.raises(NotImplementedError, match="restart"):
            driver._run_fv3_duo(start_step=7)


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
#
# 2026-08-19, merging main: the default surface gained a CMOR output
# resolution and a land diurnal-convection timescale. Neither is read on this
# lane -- it is dry and runs no physics -- so both stay DENIED and the allow
# list is unchanged. That re-review is what this hash exists to force.
_WALL_SURFACE_SHA256 = "d1c310b152e798119b0692d095e68d1fc04197f6c396a811e991abd6da47fe2b"


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
