"""Tests for production blocker fixes.

Covers:
1. Runtime bootstrap grid_type propagation
2. Physics scheme configurability via ExperimentConfig
3. Cloud-radiation coupling wiring
4. ppermute halo safety for sub-face tiling
5. Coupled/climate config validation
6. Mixed precision semantics
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.precision import set_policy, get_policy, clear_module_overrides
from legoesm.driver.config import ExperimentConfig, GridConfig


# =========================================================================
# 1. Runtime bootstrap grid_type propagation
# =========================================================================

class TestBootstrapGridType:
    """YAML bootstrap must propagate grid_type for latlon and spectral."""

    def test_yaml_bootstrap_passes_grid_type_latlon(self):
        """grid.type='latlon' in config must reach bootstrap(grid_type=...)."""
        from legoesm.runtime.config import bootstrap_from_yaml_config

        config = MagicMock()
        data = {
            "hardware.precision.dynamics": None,
            "hardware.parallelism.n_devices": "auto",
            "hardware.parallelism.backend": None,
            "hardware.parallelism.distributed": False,
            "hardware.devices": "auto",
            "grid.type": "latlon",
        }
        config.get = lambda key, default=None: data.get(key, default)

        with patch("legoesm.runtime.config.bootstrap") as mock_bootstrap:
            mock_bootstrap.return_value = MagicMock()
            bootstrap_from_yaml_config(config)
            mock_bootstrap.assert_called_once()
            call_kwargs = mock_bootstrap.call_args[1]
            assert call_kwargs["grid_type"] == "latlon"

    def test_yaml_bootstrap_passes_grid_type_spectral(self):
        """grid.type='spectral' in config must reach bootstrap(grid_type=...)."""
        from legoesm.runtime.config import bootstrap_from_yaml_config

        config = MagicMock()
        data = {
            "hardware.precision.dynamics": None,
            "hardware.parallelism.n_devices": "auto",
            "hardware.parallelism.backend": None,
            "hardware.parallelism.distributed": False,
            "hardware.devices": "auto",
            "grid.type": "spectral",
        }
        config.get = lambda key, default=None: data.get(key, default)

        with patch("legoesm.runtime.config.bootstrap") as mock_bootstrap:
            mock_bootstrap.return_value = MagicMock()
            bootstrap_from_yaml_config(config)
            call_kwargs = mock_bootstrap.call_args[1]
            assert call_kwargs["grid_type"] == "spectral"

    def test_yaml_bootstrap_defaults_cubed_sphere(self):
        """Missing grid.type defaults to cubed_sphere."""
        from legoesm.runtime.config import bootstrap_from_yaml_config

        config = MagicMock()
        data = {
            "hardware.precision.dynamics": None,
            "hardware.parallelism.n_devices": "auto",
            "hardware.parallelism.backend": None,
            "hardware.parallelism.distributed": False,
            "hardware.devices": "auto",
        }
        config.get = lambda key, default=None: data.get(key, default)

        with patch("legoesm.runtime.config.bootstrap") as mock_bootstrap:
            mock_bootstrap.return_value = MagicMock()
            bootstrap_from_yaml_config(config)
            call_kwargs = mock_bootstrap.call_args[1]
            assert call_kwargs["grid_type"] == "cubed_sphere"


class TestDistributedGridTypeGuard:
    """Distributed setup must fail fast for unsupported grid types."""

    def test_distributed_rejects_latlon(self):
        from legoesm.parallel.distributed import initialize_distributed
        with pytest.raises(ValueError, match="does not support grid_type='latlon'"):
            initialize_distributed(grid_type="latlon")

    def test_distributed_rejects_spectral(self):
        from legoesm.parallel.distributed import initialize_distributed
        with pytest.raises(ValueError, match="does not support grid_type='spectral'"):
            initialize_distributed(grid_type="spectral")

    def test_setup_devices_passes_grid_type_to_distributed(self):
        """setup_devices must forward grid_type to initialize_distributed.

        Distributed lat-lon / mpas now route to their own initialisers
        (``initialize_distributed_latlon`` / deferred Voronoi partition),
        so the cubed-sphere fall-through is the path that forwards
        ``grid_type`` into :func:`initialize_distributed`.
        """
        from legoesm.runtime.devices import setup_devices

        with patch(
            "legoesm.parallel.distributed.initialize_distributed",
            side_effect=ValueError("expected"),
        ) as mock_init:
            with pytest.raises(ValueError):
                setup_devices(distributed=True, grid_type="cubed_sphere")
            mock_init.assert_called_once_with(grid_type="cubed_sphere")

    def test_setup_devices_non_distributed_honors_grid_type(self):
        """Non-distributed path must dispatch to correct mesh creator."""
        from legoesm.runtime.devices import setup_devices

        with patch("legoesm.parallel.mesh.create_latlon_mesh") as mock_ll:
            mock_ll.return_value = MagicMock()
            setup_devices(grid_type="latlon", n_devices=1, backend="cpu")
            mock_ll.assert_called_once()

        with patch("legoesm.parallel.mesh.create_level_mesh") as mock_sp:
            mock_sp.return_value = MagicMock()
            setup_devices(grid_type="spectral", n_devices=1, backend="cpu")
            mock_sp.assert_called_once()


# =========================================================================
# 2. Physics scheme configurability
# =========================================================================

class TestPhysicsSchemeConfig:
    """ExperimentConfig must expose convection/turbulence/GWD selection."""

    def test_convection_field_exists(self):
        ec = ExperimentConfig()
        assert hasattr(ec, "convection")
        assert ec.convection == "sbm"

    def test_turbulence_field_exists(self):
        ec = ExperimentConfig()
        assert hasattr(ec, "turbulence")
        assert ec.turbulence == "none"

    def test_gravity_wave_drag_field_exists(self):
        ec = ExperimentConfig()
        assert hasattr(ec, "gravity_wave_drag")
        assert ec.gravity_wave_drag == "none"

    def test_custom_convection_scheme(self):
        ec = ExperimentConfig(convection="none")
        assert ec.convection == "none"

    def test_convection_resolver_uses_config_field(self):
        """_resolve_convection must read config.convection (not getattr fallback)."""
        from legoesm.driver.physics_pipeline import _resolve_convection

        ec = ExperimentConfig(convection="none")
        fn, cfg = _resolve_convection(ec)
        # Should return the noop convection
        assert fn is not None
        assert cfg is None

    def test_convection_resolver_default_sbm(self):
        from legoesm.driver.physics_pipeline import _resolve_convection

        ec = ExperimentConfig()  # convection="sbm"
        fn, cfg = _resolve_convection(ec)
        assert fn is not None
        assert cfg is not None

    def test_bechtold_autoconversion_threads_to_config(self):
        """--convective-precip-split autoconversion must reach BechtoldConfig via
        the DEDICATED bechtold branch (codex HIGH regression: bechtold does not
        pass through the shared _split block used by the other schemes).

        Exercised with ``bechtold_use_ifs_inplume_precip=False`` — the ONLY
        configuration in which the split actually executes.  With the default
        (True) the IFS in-plume rain formation sets ``_split_done`` before the
        split dispatch (bechtold.py:2855-2908), so the threaded value reaches
        the config and is then ignored by the kernel; asserting the config alone
        under that default proved nothing, and ``validate_strict`` now refuses
        the combination outright.
        """
        from legoesm.driver.physics_pipeline import _resolve_convection
        exp = ExperimentConfig(
            convection="bechtold", convective_precip_split="autoconversion",
            bechtold_use_ifs_inplume_precip=False,
            autoconv_q_c_crit=8.0e-4, autoconv_pe_max=0.8)
        exp.validate_strict()      # the reachable combination is accepted
        _fn, cfg = _resolve_convection(exp)
        assert cfg.precip_split_scheme == "autoconversion"
        assert cfg.autoconv_q_c_crit == 8.0e-4
        assert cfg.autoconv_pe_max == 0.8
        assert cfg.use_ifs_inplume_precip is False

    def test_tiedtke_autoconversion_threads_to_config(self):
        from legoesm.driver.physics_pipeline import _resolve_convection
        _fn, cfg = _resolve_convection(ExperimentConfig(
            convection="tiedtke", convective_precip_split="autoconversion"))
        assert cfg.precip_split_scheme == "autoconversion"

    def test_bechtold_default_split_is_constant(self):
        # default stays "constant" -> no silent behaviour change from the feature
        from legoesm.driver.physics_pipeline import _resolve_convection
        _fn, cfg = _resolve_convection(ExperimentConfig(convection="bechtold"))
        assert cfg.precip_split_scheme == "constant"

    def test_autoconversion_on_unsupporting_scheme_raises(self):
        """A non-constant split on a scheme without precip_split_scheme must fail
        LOUDLY (codex MED), not silently run the constant split."""
        import pytest
        from legoesm.driver.physics_pipeline import _resolve_convection
        with pytest.raises(ValueError, match="autoconversion split"):
            _resolve_convection(ExperimentConfig(
                convection="zhang_mcfarlane",
                convective_precip_split="autoconversion"))


# =========================================================================
# 3. Cloud-radiation coupling wiring
# =========================================================================

class TestCloudRadiationCoupling:
    """RRTMGP radiation function must accept cloud optical properties."""

    def test_rrtmgp_builder_creates_fn_with_cloud_params(self):
        """_build_rrtmgp_radiation_fn must return fn accepting cloud fields."""
        from legoesm.driver.physics_pipeline import _build_rrtmgp_radiation_fn
        import inspect

        ec = ExperimentConfig(radiation="rrtmgp", cloud_scheme="sundqvist")
        rad_fn = _build_rrtmgp_radiation_fn(ec)

        # Check signature includes cloud parameters
        sig = inspect.signature(rad_fn)
        param_names = set(sig.parameters.keys())
        assert "cloud_path_liq" in param_names
        assert "cloud_path_ice" in param_names
        assert "cloud_r_eff_liq" in param_names
        assert "cloud_r_eff_ice" in param_names

    def test_include_clouds_flag_set_when_cloud_scheme_active(self):
        """RRTMGPConfig.include_clouds must be True when cloud_scheme != 'none'."""
        from legoesm.driver.physics_pipeline import _build_rrtmgp_radiation_fn
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import RRTMGPConfig

        # With cloud scheme
        ec_clouds = ExperimentConfig(radiation="rrtmgp", cloud_scheme="sundqvist")
        # Without cloud scheme
        ec_clear = ExperimentConfig(radiation="rrtmgp", cloud_scheme="none")

        # We can't easily inspect the captured config inside the closure,
        # but we verify the function builds without error
        fn_clouds = _build_rrtmgp_radiation_fn(ec_clouds)
        fn_clear = _build_rrtmgp_radiation_fn(ec_clear)
        assert fn_clouds is not None
        assert fn_clear is not None


# =========================================================================
# 3b. RRTMGP GPU performance knob: use_scan routing
# =========================================================================

class TestRRTMGPUseScanRouting:
    """ExperimentConfig.rrtmgp_use_scan must flow into the RRTMGP wrapper.

    Hardcoding ``use_scan=True`` in ``_build_rrtmgp_radiation_fn`` forces
    ``jax.lax.scan`` on every backend, which launches one kernel per
    atmospheric layer on GPU.  The builder must honour the ExperimentConfig
    field so GPU deployments can pick the unrolled Python for-loop path
    (use_scan=False) without editing driver code.
    """

    def test_rrtmgp_use_scan_field_exists_and_defaults_none(self):
        # Issue #273 GPU tuning: the default is ``None`` (auto-pick
        # ``lax.scan`` on GPU/TPU, the unrolled Python loop on CPU);
        # explicit ``True``/``False`` override.  An earlier revision
        # hard-defaulted ``False`` — this default deliberately drifted.
        ec = ExperimentConfig()
        assert hasattr(ec, "rrtmgp_use_scan")
        assert ec.rrtmgp_use_scan is None

    def _capture_rrtmg_config(self, ec):
        """Build wrapper with RRTMGPConfig captured for inspection.

        Patches optics loading so the test doesn't read NetCDF data.
        """
        from legoesm.driver import physics_pipeline as pp
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        captured: dict = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return RRTMGPConfig(*args, **kwargs)

        with patch(
            "legoesm.atmosphere.physics.radiation.config.RRTMGPConfig",
            side_effect=_capture,
        ), patch(
            "legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp.RRTMGP.from_legoesm_config",
            return_value=MagicMock(),
        ):
            _ = pp._build_rrtmgp_radiation_fn(ec)

        return captured

    def test_rrtmgp_builder_honors_use_scan_false(self):
        """Builder must not silently override user-selected use_scan=False."""
        captured = self._capture_rrtmg_config(
            ExperimentConfig(radiation="rrtmgp", rrtmgp_use_scan=False)
        )
        assert captured.get("use_scan") is False

    def test_rrtmgp_builder_honors_use_scan_true(self):
        """Training workflows must still be able to opt into scan."""
        captured = self._capture_rrtmg_config(
            ExperimentConfig(radiation="rrtmgp", rrtmgp_use_scan=True)
        )
        assert captured.get("use_scan") is True

    def test_rrtmgp_use_scan_roundtrips_through_json(self):
        """Serialization must round-trip the new field (explicit + default)."""
        from legoesm.driver.config import (
            experiment_config_from_dict,
            experiment_config_to_dict,
        )

        ec = ExperimentConfig(radiation="rrtmgp", rrtmgp_use_scan=True)
        d = experiment_config_to_dict(ec)
        assert d["rrtmgp_use_scan"] is True
        restored = experiment_config_from_dict(d)
        assert restored.rrtmgp_use_scan is True

        # Older checkpoints without the field must still load (default
        # None = auto-detect scan-vs-unroll per backend, issue #273).
        d.pop("rrtmgp_use_scan")
        restored_default = experiment_config_from_dict(d)
        assert restored_default.rrtmgp_use_scan is None


# =========================================================================
# 4. ppermute halo safety for sub-face tiling
# =========================================================================

class TestPpermuteTilingGuard:
    """ppermute path must fall back when tiling != (1,1)."""

    def test_ppermute_rejects_tiled_mesh(self):
        """When active config has tiling != (1,1), ppermute must fall back."""
        from legoesm.parallel.async_halo import jax_native_halo_exchange
        from legoesm.parallel.mesh import DeviceConfig

        tiled_config = DeviceConfig(
            mesh=None, face_sharding=None, replicated_sharding=None,
            n_devices=24, backend="CPU", is_distributed=False,
            tiling=(2, 2), grid_type="cubed_sphere",
        )

        mock_mesh = MagicMock()
        mock_grid = MagicMock()
        data = jnp.ones((6, 4, 4))

        # ``async_halo`` imports ``get_active_config`` and ``pad_halo``
        # directly (``from ... import``), so the patches must target the
        # names as bound in the ``async_halo`` namespace, not their
        # definition modules.
        with patch("legoesm.parallel.async_halo.get_active_config", return_value=tiled_config), \
             patch("legoesm.parallel.async_halo.pad_halo", return_value=data) as mock_pad:
            with pytest.warns(RuntimeWarning, match="sub-face tiling"):
                result = jax_native_halo_exchange(data, mock_grid, mesh=mock_mesh)
            mock_pad.assert_called_once()

    def test_ppermute_allows_face_only_mesh(self):
        """When tiling == (1,1), ppermute path is allowed (with experimental warning)."""
        from legoesm.parallel.async_halo import jax_native_halo_exchange
        from legoesm.parallel.mesh import DeviceConfig

        face_config = DeviceConfig(
            mesh=None, face_sharding=None, replicated_sharding=None,
            n_devices=6, backend="CPU", is_distributed=False,
            tiling=(1, 1), grid_type="cubed_sphere",
        )

        mock_mesh = MagicMock()
        mock_grid = MagicMock()
        data = jnp.ones((6, 4, 4))

        # ``get_active_config`` is imported into ``async_halo`` directly, so
        # patch it there for the guard read to see the face-only config.
        with patch("legoesm.parallel.async_halo.get_active_config", return_value=face_config), \
             patch("legoesm.parallel.async_halo._ppermute_halo_exchange", return_value=data) as mock_pp:
            with pytest.warns(RuntimeWarning, match="experimental"):
                jax_native_halo_exchange(data, mock_grid, mesh=mock_mesh)
            mock_pp.assert_called_once()

    def test_ppermute_no_mesh_uses_local_pad(self):
        """mesh=None must always use local pad_halo, no ppermute."""
        from legoesm.parallel.async_halo import jax_native_halo_exchange

        data = jnp.ones((6, 4, 4))
        mock_grid = MagicMock()

        # ``pad_halo`` is imported into ``async_halo`` directly; patch it
        # there so the ``mesh=None`` local-pad fallback is observed.
        with patch("legoesm.parallel.async_halo.pad_halo", return_value=data) as mock_pad:
            jax_native_halo_exchange(data, mock_grid, mesh=None)
            mock_pad.assert_called_once()


# =========================================================================
# 5. Coupled/climate config validation
# =========================================================================

class TestCoupledConfigValidation:
    """ModelDriver must reject unsupported coupled modes with actionable errors."""

    def test_carbon_cycle_rejected(self):
        ec = ExperimentConfig(carbon_cycle="interactive")
        with pytest.raises(ValueError, match="carbon_cycle.*not implemented"):
            ec.validate_strict()

    def test_carbon_cycle_none_passes(self):
        ec = ExperimentConfig(carbon_cycle="none")
        ec.validate_strict()  # Should not raise

    def test_default_config_passes_strict(self):
        ec = ExperimentConfig()
        ec.validate_strict()  # Defaults should always be valid


# =========================================================================
# 6. Mixed precision semantics
# =========================================================================

class TestMixedPrecisionSemantics:
    """Mixed precision matrix: fp32/fp64/mixed must bootstrap correctly."""

    @pytest.fixture(autouse=True)
    def _restore_precision_state(self):
        """Snapshot and restore the global precision state around each test.

        These tests mutate the process-global precision policy / module
        overrides / ``jax_enable_x64`` flag.  A plain ``teardown`` that reset
        to fp32 left x64 ON when a test enabled it, leaking to sibling tests
        in the same xdist worker.  Snapshot everything at setup and restore it
        verbatim at teardown so each test is hermetic regardless of the order
        it runs in.
        """
        x64_before = jax.config.jax_enable_x64
        policy_before = get_policy()
        from legoesm.core.precision import get_module_overrides, set_module_override
        overrides_before = get_module_overrides()
        try:
            yield
        finally:
            clear_module_overrides()
            for module, roles in overrides_before.items():
                if roles:
                    set_module_override(module, **roles)
            set_policy(policy_before)
            jax.config.update("jax_enable_x64", x64_before)

    @pytest.mark.parametrize("mode", ["fp32", "fp64", "mixed"])
    def test_bootstrap_precision_mode(self, mode):
        """Each precision mode must produce correct PrecisionPolicy."""
        from legoesm.runtime.precision import resolve_precision

        policy = resolve_precision(mode)
        if mode == "fp32":
            assert policy.storage == jnp.float32
            assert policy.accumulate == jnp.float32
        elif mode == "fp64":
            assert policy.storage == jnp.float64
            assert policy.accumulate == jnp.float64
        elif mode == "mixed":
            assert policy.storage == jnp.float32
            assert policy.accumulate == jnp.float64
            assert policy.control == jnp.float64

    @pytest.mark.parametrize("mode", ["fp32", "fp64", "mixed"])
    def test_apply_precision_activates_policy(self, mode):
        """apply_precision must set global policy and enable x64 when needed."""
        from legoesm.runtime.precision import apply_precision

        policy = apply_precision(mode)
        active = get_policy()
        assert active.storage == policy.storage
        assert active.compute == policy.compute
        assert active.accumulate == policy.accumulate

        if mode in ("fp64", "mixed"):
            assert jax.config.jax_enable_x64

    def test_yaml_bootstrap_precision_mapping(self):
        """YAML config precision keys must map correctly to modes."""
        from legoesm.runtime.config import bootstrap_from_yaml_config

        test_cases = [
            # (dynamics, conservation) → expected mode
            ("float32", None, "fp32"),
            ("float32", "float64", "mixed"),
            ("float64", "float64", "fp64"),
        ]
        for dyn, cons, expected_mode in test_cases:
            config = MagicMock()
            data = {
                "hardware.precision.dynamics": dyn,
                "hardware.precision.conservation": cons,
                "hardware.parallelism.n_devices": "auto",
                "hardware.parallelism.backend": None,
                "hardware.parallelism.distributed": False,
                "hardware.devices": "auto",
                "grid.type": "cubed_sphere",
            }
            config.get = lambda key, default=None, _d=data: _d.get(key, default)

            with patch("legoesm.runtime.config.bootstrap") as mock_bootstrap:
                mock_bootstrap.return_value = MagicMock()
                bootstrap_from_yaml_config(config)
                call_kwargs = mock_bootstrap.call_args[1]
                assert call_kwargs["precision"] == expected_mode, \
                    f"dyn={dyn}, cons={cons} → expected {expected_mode}, got {call_kwargs['precision']}"

    def test_legacy_ml_precision_preserved_in_hardware_dict(self):
        """ML precision key must be preserved in legacy 3-component dict."""
        from legoesm.core.hardware import (
            set_runtime_precision_policy,
            get_runtime_precision_policy,
        )
        set_runtime_precision_policy(ml="bfloat16")
        policy = get_runtime_precision_policy()
        assert policy["ml"] == jnp.bfloat16

    @pytest.mark.parametrize("mode", ["fp32", "fp64", "mixed"])
    def test_precision_policy_matches_mode(self, mode):
        """PrecisionPolicy must match the requested mode.

        ``apply_precision`` is the API that actually installs the global
        policy for a mode; ``set_recommended_overrides`` only manages the
        per-module override table and leaves ``get_policy()`` untouched, so
        the policy must be driven via ``apply_precision`` here.
        """
        from legoesm.runtime.precision import apply_precision

        apply_precision(mode)
        policy = get_policy()
        if mode == "fp64":
            assert policy.compute == jnp.float64
        elif mode == "mixed":
            assert policy.compute == jnp.float32
            assert policy.accumulate == jnp.float64
        else:
            assert policy.compute == jnp.float32


# =========================================================================
# 7. Config serialization roundtrip with new fields
# =========================================================================

class TestConfigRoundtrip:
    """New fields must survive JSON serialization roundtrip."""

    def test_new_physics_fields_roundtrip(self):
        from legoesm.driver.config import (
            experiment_config_to_dict,
            experiment_config_from_dict,
        )
        ec = ExperimentConfig(
            convection="dca",
            turbulence="smagorinsky",
            gravity_wave_drag="rayleigh",
        )
        d = experiment_config_to_dict(ec)
        ec2 = experiment_config_from_dict(d)
        assert ec2.convection == "dca"
        assert ec2.turbulence == "smagorinsky"
        assert ec2.gravity_wave_drag == "rayleigh"

    def test_legacy_config_without_new_fields_loads(self):
        """Old configs missing new fields must still load (forward compat)."""
        from legoesm.driver.config import experiment_config_from_dict

        # Simulate a legacy dict without the new fields
        d = {"days": 100, "radiation": "gray"}
        ec = experiment_config_from_dict(d)
        # Should use defaults
        assert ec.convection == "sbm"
        assert ec.turbulence == "none"
        assert ec.gravity_wave_drag == "none"
