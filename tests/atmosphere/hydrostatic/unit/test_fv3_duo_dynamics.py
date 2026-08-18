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

    @pytest.mark.parametrize("bad", [
        dict(radiation="gray"),
        dict(convection="sbm"),
        dict(microphysics="sundqvist"),
        dict(turbulence="louis"),
        dict(precision="fp32"),
        dict(held_suarez_forcing=True),
        dict(distributed=True),
    ])
    def test_slice1_refusals_fire(self, bad):
        """Every refusal raises BEFORE any (expensive) duo grid build."""
        from legoesm.driver.component_factory import create_atmosphere_dycore
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        cfg = _fv3_duo_config(**bad)
        with pytest.raises(ValueError, match="fv3_duo"):
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

    def test_restart_refused(self, tmp_path):
        from legoesm.driver.model_driver import ModelDriver
        cfg = _fv3_duo_config(output_dir=str(tmp_path))
        driver = ModelDriver(cfg, output_dir=tmp_path)
        driver.setup()
        with pytest.raises(NotImplementedError, match="restart"):
            driver._run_fv3_duo(start_step=7)
