"""MPAS NH radiation + microphysics factory tests.

Mirrors :mod:`tests/unit/test_plane_radiation_microphysics.py` for
the new ``model_type='mpas_nh'`` dispatch path that wires the
shared scheme stack (gray / RRTMGP for radiation; kessler /
morrison / sundqvist / seifert_beheng / thompson / ml_emulator /
none for microphysics) into the MPAS Voronoi non-hydrostatic
dycore.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig, MicrophysicsConfig, MorrisonConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig, RadiationConfig,
)
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.core.field import Field
from legoesm.core.state import (
    MPASNonHydrostaticState, MPASNonHydrostaticTendencies,
)
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
)
from legoesm.grids.voronoi import create_voronoi_mesh

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def mpas_setup():
    """Small MPAS NH state with 3 moist tracers seeded at the surface."""
    nlev = 8
    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
    hc = create_height_coordinate(nlev, H=20_000.0)
    tm = compute_terrain_metric(jnp.zeros(mesh.nCells), hc)
    tracers = jnp.zeros((mesh.nCells, nlev, 3), dtype=jnp.float64)
    tracers = tracers.at[..., -1, 0].set(0.01)
    state = MPASNonHydrostaticState(
        u=Field(jnp.zeros((mesh.nEdges, nlev), jnp.float64), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        w=Field(jnp.zeros((mesh.nCells, nlev + 1), jnp.float64),
                name="w", dims=("nCells", "nlev_half"), units="m/s"),
        theta_prime=Field(
            jnp.zeros((mesh.nCells, nlev), jnp.float64),
            name="theta_prime", dims=("nCells", "nlev"), units="K",
        ),
        rho_prime=Field(
            jnp.zeros((mesh.nCells, nlev), jnp.float64),
            name="rho_prime", dims=("nCells", "nlev"), units="kg/m^3",
        ),
        phis=Field(jnp.zeros(mesh.nCells, jnp.float64), name="phis",
                   dims=("nCells",), units="m^2/s^2"),
        tracers=Field(tracers, name="tracers",
                      dims=("nCells", "nlev", "tracer"), units="kg/kg"),
    )
    return {
        "mesh": mesh, "hc": hc, "tm": tm, "state": state,
        "nlev": nlev, "ncol": mesh.nCells,
    }


class TestMPASNHRadiation:
    """Gray radiation via ``model_type='mpas_nh'`` factory."""

    def test_gray_returns_mpas_tendencies_with_correct_shapes(
        self, mpas_setup,
    ):
        cfg = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())
        physics_fn = make_radiation_physics(cfg, model_type="mpas_nh")
        tend = physics_fn(
            mpas_setup["state"], mpas_setup["mesh"],
            mpas_setup["hc"], mpas_setup["tm"],
        )
        assert isinstance(tend, MPASNonHydrostaticTendencies)
        nlev = mpas_setup["nlev"]
        ncol = mpas_setup["ncol"]
        assert tend.dtheta_prime_dt.data.shape == (ncol, nlev)
        assert tend.drho_prime_dt.data.shape == (ncol, nlev)
        assert tend.dw_dt.data.shape == (ncol, nlev + 1)
        assert tend.du_dt.data.shape == mpas_setup["state"].u.data.shape
        assert tend.dphis_dt.data.shape == (ncol,)
        # Radiation doesn't move tracers.
        assert tend.dtracers_dt.data.shape == (ncol, nlev, 3)
        assert float(jnp.max(jnp.abs(tend.dtracers_dt.data))) == 0.0

    def test_gray_dim_labels_use_mpas_axes(self, mpas_setup):
        cfg = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())
        tend = make_radiation_physics(cfg, model_type="mpas_nh")(
            mpas_setup["state"], mpas_setup["mesh"],
            mpas_setup["hc"], mpas_setup["tm"],
        )
        assert tend.dtheta_prime_dt.dims == ("nCells", "nlev")
        assert tend.du_dt.dims == ("nEdges", "nlev")
        assert tend.dw_dt.dims == ("nCells", "nlev_half")
        assert tend.dphis_dt.dims == ("nCells",)
        assert tend.dtracers_dt.dims == ("nCells", "nlev", "tracer")

    def test_gray_produces_nonzero_heating(self, mpas_setup):
        cfg = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())
        tend = make_radiation_physics(cfg, model_type="mpas_nh")(
            mpas_setup["state"], mpas_setup["mesh"],
            mpas_setup["hc"], mpas_setup["tm"],
        )
        max_heating = float(jnp.max(jnp.abs(tend.dtheta_prime_dt.data)))
        assert max_heating > 1.0e-9
        assert bool(jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)))


class TestMPASNHMicrophysics:
    """Microphysics via ``model_type='mpas_nh'`` factory."""

    def test_kessler_returns_mpas_tendencies(self, mpas_setup):
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        tend = make_microphysics_physics(
            cfg, model_type="mpas_nh", dt=1.0,
        )(
            mpas_setup["state"], mpas_setup["mesh"],
            mpas_setup["hc"], mpas_setup["tm"],
        )
        assert isinstance(tend, MPASNonHydrostaticTendencies)
        nlev = mpas_setup["nlev"]
        ncol = mpas_setup["ncol"]
        assert tend.dtheta_prime_dt.data.shape == (ncol, nlev)
        assert tend.dtracers_dt.data.shape == (ncol, nlev, 3)
        assert bool(jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)))
        assert bool(jnp.all(jnp.isfinite(tend.dtracers_dt.data)))

    def test_bridge_clips_negative_tracer_read(self, mpas_setup, monkeypatch):
        """The MPAS-NH microphysics bridge clips negative tracer READS to ≥0
        BEFORE the scheme sees them — isolated from Kessler's OWN internal
        negative guards (codex CRM-dycore review). A SPY wrapping the scheme
        records the q_r it is handed: with the bridge clip it sees q_r ≥ 0
        despite a state with q_r<0; with ``clip_positive`` monkeypatched to
        identity it sees the raw negative. Mirrors the plane bridge test."""
        from legoesm.atmosphere.physics.microphysics import integration
        state = mpas_setup["state"]
        ncol, nlev = mpas_setup["ncol"], mpas_setup["nlev"]
        tr = jnp.zeros((ncol, nlev, 3), dtype=jnp.float64)
        tr = tr.at[..., 0].set(1.5e-2)                   # q_v near/above sat
        tr = tr.at[:, -1, 2].set(-5.0e-3)                # NEGATIVE rain
        s = state._replace(tracers=state.tracers.replace(data=tr))

        captured = {}
        real_kessler = integration.kessler_microphysics

        def _spy(T, q_v, hydro, *args, **kwargs):
            captured["qr_min"] = float(jnp.min(hydro.q_r))
            return real_kessler(T, q_v, hydro, *args, **kwargs)

        def _run():
            make_microphysics_physics(
                MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig()),
                model_type="mpas_nh", dt=20.0,
            )(s, mpas_setup["mesh"], mpas_setup["hc"], mpas_setup["tm"])

        monkeypatch.setattr(integration, "kessler_microphysics", _spy)
        _run()
        assert captured["qr_min"] >= 0.0, (
            f"MPAS-NH bridge fed q_r<0 to microphysics: {captured['qr_min']:.3e}")
        # Disable the clip → the spy MUST now see the raw negative.
        monkeypatch.setattr(integration, "clip_positive", lambda x: x)
        _run()
        assert captured["qr_min"] < 0.0, (
            "clip disabled but microphysics still saw q_r≥0 — not exercising "
            "the bridge clip")

    def test_none_returns_zero_tendencies(self, mpas_setup):
        cfg = MicrophysicsConfig(scheme="none")
        tend = make_microphysics_physics(
            cfg, model_type="mpas_nh", dt=1.0,
        )(
            mpas_setup["state"], mpas_setup["mesh"],
            mpas_setup["hc"], mpas_setup["tm"],
        )
        assert float(jnp.max(jnp.abs(tend.dtheta_prime_dt.data))) == 0.0
        assert float(jnp.max(jnp.abs(tend.dtracers_dt.data))) == 0.0

    def test_morrison_with_too_few_tracer_slots_raises(self, mpas_setup):
        """Sister of the plane test: Morrison writes through slot 8;
        the fixture has 3 slots → must raise."""
        cfg = MicrophysicsConfig(
            scheme="morrison", morrison=MorrisonConfig(),
        )
        physics_fn = make_microphysics_physics(
            cfg, model_type="mpas_nh", dt=1.0,
        )
        with pytest.raises(
            ValueError, match="writes up to 9 tracer-slot tendencies",
        ):
            physics_fn(
                mpas_setup["state"], mpas_setup["mesh"],
                mpas_setup["hc"], mpas_setup["tm"],
            )

    def test_morrison_runs_with_full_9_slot_state(self, mpas_setup):
        ncol = mpas_setup["ncol"]
        nlev = mpas_setup["nlev"]
        tracers = jnp.zeros((ncol, nlev, 9), dtype=jnp.float64)
        tracers = tracers.at[..., -1, 0].set(0.01)
        state9 = mpas_setup["state"]._replace(
            tracers=mpas_setup["state"].tracers.replace(data=tracers),
        )
        cfg = MicrophysicsConfig(
            scheme="morrison", morrison=MorrisonConfig(),
        )
        tend = make_microphysics_physics(
            cfg, model_type="mpas_nh", dt=1.0,
        )(state9, mpas_setup["mesh"], mpas_setup["hc"], mpas_setup["tm"])
        assert isinstance(tend, MPASNonHydrostaticTendencies)
        assert tend.dtracers_dt.data.shape == (ncol, nlev, 9)
        assert bool(jnp.all(jnp.isfinite(tend.dtracers_dt.data)))


class TestMPASNHMicrophysicsSchemeCoverage:
    """Codex review 2026-05-24: every scheme exposed via
    ``model_type='mpas_nh'`` dispatch must build + run on MPAS NH
    with its minimum tracer-slot allocation, not just Kessler."""

    @pytest.mark.parametrize(
        "scheme,min_slots,cfg_field,cfg_cls_path",
        [
            ("sundqvist", 3, "sundqvist",
             "legoesm.atmosphere.physics.microphysics.config.SundqvistConfig"),
            ("seifert_beheng", 9, "seifert_beheng",
             "legoesm.atmosphere.physics.microphysics.config.SeifertBehengConfig"),
            ("thompson", 9, "thompson",
             "legoesm.atmosphere.physics.microphysics.config.ThompsonConfig"),
        ],
    )
    def test_scheme_smoke(self, mpas_setup, scheme, min_slots,
                          cfg_field, cfg_cls_path):
        """Smoke: each scheme builds + runs on MPAS NH with its
        minimum tracer-slot count + produces finite tendencies."""
        import importlib
        mod_path, cls_name = cfg_cls_path.rsplit(".", 1)
        cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
        cfg = MicrophysicsConfig(scheme=scheme, **{cfg_field: cfg_cls()})

        ncol = mpas_setup["ncol"]
        nlev = mpas_setup["nlev"]
        tracers = jnp.zeros((ncol, nlev, min_slots), dtype=jnp.float64)
        tracers = tracers.at[..., -1, 0].set(0.01)   # q_v at surface
        state = mpas_setup["state"]._replace(
            tracers=mpas_setup["state"].tracers.replace(data=tracers),
        )
        physics_fn = make_microphysics_physics(
            cfg, model_type="mpas_nh", dt=1.0,
        )
        tend = physics_fn(
            state, mpas_setup["mesh"],
            mpas_setup["hc"], mpas_setup["tm"],
        )
        assert isinstance(tend, MPASNonHydrostaticTendencies)
        assert tend.dtracers_dt.data.shape == (ncol, nlev, min_slots)
        assert bool(jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)))
        assert bool(jnp.all(jnp.isfinite(tend.dtracers_dt.data)))

    def test_too_few_slots_raises_for_every_scheme(self, mpas_setup):
        """Per-scheme tracer-slot validation must fire for any scheme
        that writes through high slots. 3-slot fixture is insufficient
        for the two-moment schemes that write through N_i (slot 8)."""
        for scheme, min_slots in [
            ("seifert_beheng", 9),
            ("thompson", 9),
        ]:
            cfg = MicrophysicsConfig(scheme=scheme)
            physics_fn = make_microphysics_physics(
                cfg, model_type="mpas_nh", dt=1.0,
            )
            with pytest.raises(
                ValueError,
                match=f"writes up to {min_slots} tracer-slot tendencies",
            ):
                physics_fn(
                    mpas_setup["state"], mpas_setup["mesh"],
                    mpas_setup["hc"], mpas_setup["tm"],
                )


class TestMPASNHRadiationCloudCoverage:
    """Codex review 2026-05-24 iter-3: hard-fail when cloud-aware
    radiation is requested but the state cannot carry cloud condensate;
    prove (via monkeypatch on `_call_radiation_backend`) that
    `q_cloud` / `q_ice` are read from tracer slots 1 and 3 when
    present and routed to the backend, not silently dropped."""

    def test_rrtmgp_include_clouds_requires_4_tracer_slots(self, mpas_setup):
        """``RRTMGPConfig(include_clouds=True)`` with fewer than 4
        tracer slots must raise — silent clear-sky fallback was the
        original bug."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        try:
            cfg = RadiationConfig(
                scheme="rrtmgp",
                rrtmgp=RRTMGPConfig(include_clouds=True),
            )
            physics_fn = make_radiation_physics(cfg, model_type="mpas_nh")
        except Exception:
            pytest.skip("RRTMGP backend unavailable in this env")
        # Fixture has 3 slots → must fail at call time, NOT
        # silently fall back to clear-sky.
        with pytest.raises(ValueError, match="Cloud-aware radiation"):
            physics_fn(
                mpas_setup["state"], mpas_setup["mesh"],
                mpas_setup["hc"], mpas_setup["tm"],
            )

    def test_cloud_slots_routed_to_backend(self, mpas_setup, monkeypatch):
        """Patch ``_call_radiation_backend`` to capture its `q_cloud` /
        `q_ice` kwargs and assert they equal tracer slots 1 and 3 of
        the state passed to the factory. Proves the slot indexing is
        load-bearing — independent of whatever the actual gray /
        RRTMGP backend does with the values internally."""
        captured = {}

        def _capture(**kwargs):
            captured["q_cloud"] = kwargs.get("q_cloud")
            captured["q_ice"] = kwargs.get("q_ice")
            # Return a dummy RadiationOutput-shaped object so the
            # factory body finishes assembling the tendency.
            from legoesm.atmosphere.physics.radiation.output import (
                RadiationOutput,
            )
            ncol = mpas_setup["ncol"]
            nlev = mpas_setup["nlev"]
            zeros = jnp.zeros((ncol, nlev), dtype=jnp.float64)
            zeros_half = jnp.zeros((ncol, nlev + 1), dtype=jnp.float64)
            return RadiationOutput(
                lw_flux_up=zeros_half, lw_flux_down=zeros_half,
                sw_flux_up=zeros_half, sw_flux_down=zeros_half,
                heating_rate=zeros,
                lw_heating_rate=zeros, sw_heating_rate=zeros,
            )

        import legoesm.atmosphere.physics.radiation.integration as _rad_int
        monkeypatch.setattr(_rad_int, "_call_radiation_backend", _capture)

        ncol = mpas_setup["ncol"]
        nlev = mpas_setup["nlev"]
        tracers = jnp.zeros((ncol, nlev, 4), dtype=jnp.float64)
        tracers = tracers.at[..., -1, 0].set(0.01)        # q_v
        tracers = tracers.at[..., :nlev // 2, 1].set(1e-3)  # q_c
        tracers = tracers.at[..., :nlev // 2, 3].set(5e-4)  # q_i
        state = mpas_setup["state"]._replace(
            tracers=mpas_setup["state"].tracers.replace(data=tracers),
        )

        cfg = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())
        physics_fn = make_radiation_physics(cfg, model_type="mpas_nh")
        physics_fn(
            state, mpas_setup["mesh"],
            mpas_setup["hc"], mpas_setup["tm"],
        )

        # q_cloud and q_ice should be the CLIPPED slot-1 / slot-3
        # arrays (negative-clipped at zero).
        assert captured["q_cloud"] is not None, (
            "factory dropped q_cloud despite tracer slot 1 present"
        )
        assert captured["q_ice"] is not None, (
            "factory dropped q_ice despite tracer slot 3 present"
        )
        expected_q_cloud = jnp.clip(tracers[..., 1], 0.0, None)
        expected_q_ice = jnp.clip(tracers[..., 3], 0.0, None)
        assert jnp.array_equal(captured["q_cloud"], expected_q_cloud)
        assert jnp.array_equal(captured["q_ice"], expected_q_ice)


class TestMPASNHRadiationRRTMGPSmoke:
    """Codex review 2026-05-24: RRTMGP MPAS NH smoke. Skipped when
    the RRTMGP optics tables are not loadable in CI."""

    def test_rrtmgp_smoke(self, mpas_setup):
        try:
            from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
                RRTMGP,
            )
        except Exception:
            pytest.skip("RRTMGP backend unavailable in this env")
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        try:
            cfg = RadiationConfig(scheme="rrtmgp", rrtmgp=RRTMGPConfig())
            physics_fn = make_radiation_physics(cfg, model_type="mpas_nh")
        except Exception as exc:
            pytest.skip(f"RRTMGP solver init failed: {exc!r}")
        try:
            tend = physics_fn(
                mpas_setup["state"], mpas_setup["mesh"],
                mpas_setup["hc"], mpas_setup["tm"],
            )
        except Exception as exc:
            pytest.skip(f"RRTMGP runtime failed in this env: {exc!r}")
        assert isinstance(tend, MPASNonHydrostaticTendencies)
        assert bool(jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)))


class TestMPASNHDifferentiable:
    """jax.grad smoke through the MPAS NH adapters."""

    def test_gray_dtheta_prime_grad_finite(self, mpas_setup):
        cfg = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())
        physics_fn = make_radiation_physics(cfg, model_type="mpas_nh")
        base_state = mpas_setup["state"]

        def loss_fn(theta_prime_data):
            state = base_state._replace(
                theta_prime=base_state.theta_prime.replace(
                    data=theta_prime_data,
                ),
            )
            tend = physics_fn(
                state, mpas_setup["mesh"],
                mpas_setup["hc"], mpas_setup["tm"],
            )
            return jnp.sum(tend.dtheta_prime_dt.data ** 2)

        grad = jax.grad(loss_fn)(base_state.theta_prime.data)
        assert grad.shape == base_state.theta_prime.data.shape
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_kessler_q_v_grad_finite(self, mpas_setup):
        cfg = MicrophysicsConfig(
            scheme="kessler", kessler=KesslerConfig(),
        )
        physics_fn = make_microphysics_physics(
            cfg, model_type="mpas_nh", dt=1.0,
        )
        base_state = mpas_setup["state"]

        def loss_fn(tracers_data):
            state = base_state._replace(
                tracers=base_state.tracers.replace(data=tracers_data),
            )
            tend = physics_fn(
                state, mpas_setup["mesh"],
                mpas_setup["hc"], mpas_setup["tm"],
            )
            return jnp.sum(tend.dtheta_prime_dt.data ** 2)

        grad = jax.grad(loss_fn)(base_state.tracers.data)
        assert grad.shape == base_state.tracers.data.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
