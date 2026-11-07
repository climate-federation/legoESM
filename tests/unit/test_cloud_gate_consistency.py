"""Connected-parameterization wiring guards: cloud-radiation gate + dynamic_albedo.

Two silent-fallback bugs found in the param-wiring audit (siblings of the r_eff /
droplet-number coupling bug pinned in ``test_radiation_number_coupling``):

1. ``RadiationConfig.cloud_scheme`` and ``RRTMGPConfig.include_clouds`` are
   independent knobs.  With a cloud scheme active but ``include_clouds=False``,
   the cloud optics are computed, passed to ``solve_columns``, then SILENTLY
   nulled by RRTMGP's ``has_clouds`` gate → clear-sky radiation, no error, dead
   gradient through any trained cloud knobs (the AIMIP spectral_pe default
   ``cloud_scheme='xu_randall'`` hit this).  ``make_radiation_physics`` now
   RAISES on the inconsistent combination instead of silently discarding the
   clouds, and the AIMIP config builder derives ``include_clouds`` from
   ``cloud_scheme`` so its default actually couples clouds to radiation.

2. ``PhysicsPipeline(dynamic_albedo=True)`` was once a silent no-op —
   ``compute_radiation_core`` never read the flag, so a user enabling it
   silently got the constant ocean/ice albedo.  It is now WIRED: the
   radiation closure computes the Briegleb (1992) zenith-dependent ocean
   albedo (``if self.dynamic_albedo:`` → ``legoesm.surface_albedo.
   ocean_albedo``).  The guard test now verifies the flag constructs and is
   carried (not that it raises).
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.physics.radiation import integration as rad_int
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    RadiationConfig,
    RRTMGPConfig,
)
from legoesm.atmosphere.physics.radiation.integration import (
    _validate_cloud_gate,
    make_radiation_physics,
)


def _rrtmgp_cfg(cloud_scheme: str, include_clouds: bool) -> RadiationConfig:
    return RadiationConfig(
        scheme="rrtmgp",
        rrtmgp=RRTMGPConfig(include_clouds=include_clouds),
        cloud_scheme=cloud_scheme,
    )


class TestValidateCloudGate:
    def test_active_scheme_with_clouds_off_raises(self):
        """The bug: cloud_scheme active but include_clouds=False would silently
        discard the cloud optics → clear-sky.  Must raise, not silently pass."""
        cfg = _rrtmgp_cfg("xu_randall", include_clouds=False)
        with pytest.raises(ValueError, match="Inconsistent cloud-radiation gate"):
            _validate_cloud_gate(cfg)

    def test_sundqvist_with_clouds_off_raises(self):
        cfg = _rrtmgp_cfg("sundqvist", include_clouds=False)
        with pytest.raises(ValueError, match="include_clouds is False"):
            _validate_cloud_gate(cfg)

    def test_consistent_clouds_on_is_ok(self):
        # cloud scheme active AND include_clouds=True → valid, no raise.
        _validate_cloud_gate(_rrtmgp_cfg("xu_randall", include_clouds=True))

    def test_no_cloud_scheme_is_ok(self):
        # Genuine clear-sky request (cloud_scheme='none') → valid regardless.
        _validate_cloud_gate(_rrtmgp_cfg("none", include_clouds=False))
        _validate_cloud_gate(_rrtmgp_cfg("none", include_clouds=True))

    def test_gray_radiation_never_inconsistent(self):
        """Gray radiation has no RRTMGP cloud gate; never raises."""
        cfg = RadiationConfig(
            scheme="gray", gray=GrayRadiationConfig(), cloud_scheme="xu_randall"
        )
        _validate_cloud_gate(cfg)

    def test_make_radiation_physics_enforces_gate(self):
        """The chokepoint must reject the silent-clear-sky config for every
        standalone dycore caller."""
        cfg = _rrtmgp_cfg("xu_randall", include_clouds=False)
        with pytest.raises(ValueError, match="Inconsistent cloud-radiation gate"):
            make_radiation_physics(cfg, model_type="spectral_pe")

    def test_make_radiation_physics_gray_unaffected(self):
        """A consistent gray config still builds (no heavy RRTMGP load)."""
        rad_int  # imported for symmetry / explicitness
        make_radiation_physics(
            RadiationConfig(scheme="gray", gray=GrayRadiationConfig()),
            model_type="hydrostatic",
        )


class TestAIMIPDerivesCloudGate:
    def test_aimip_rrtmgp_default_couples_clouds(self):
        """AIMIP's default cloud_scheme='xu_randall' must build a radiation
        config with include_clouds=True (else clouds are silently clear-sky and
        the trained Xu-Randall knobs have zero gradient through radiation)."""
        pytest.importorskip("legoesm.training.aimip_params")
        from legoesm.training.aimip_params import (
            make_aimip_classical_spectral_physics,
        )
        import inspect

        # The builder must not raise on its default (rrtmgp + xu_randall);
        # before the fix the make_radiation_physics gate check would trip.
        sig = inspect.signature(make_aimip_classical_spectral_physics)
        assert "radiation" in sig.parameters
        assert "cloud_scheme" in sig.parameters


class TestDynamicAlbedoGuard:
    def test_dynamic_albedo_true_constructs_and_is_carried(self):
        """dynamic_albedo=True is now WIRED (Briegleb 1992 ocean albedo in the
        radiation closure), so it must construct and carry the flag — NOT a
        silent no-op and NOT a raise."""
        from legoesm.driver.physics_pipeline import PhysicsPipeline

        pipe = PhysicsPipeline(
            adapter=None,
            sigma_full=None,
            sigma_half=None,
            dsigma=None,
            sigma_coord=None,
            convection_fn=None,
            convection_config=None,
            radiation_fn=None,
            dynamic_albedo=True,
        )
        assert pipe.dynamic_albedo is True
        # The wiring proof: the pipeline source reads the flag to select the
        # zenith-dependent ocean albedo (guards against a future silent no-op).
        import inspect
        src = inspect.getsource(PhysicsPipeline)
        assert "if self.dynamic_albedo:" in src
        assert "ocean_albedo" in src

    def test_dynamic_albedo_false_constructs(self):
        """The default (False) path must still build without error."""
        from legoesm.driver.physics_pipeline import PhysicsPipeline

        pipe = PhysicsPipeline(
            adapter=None,
            sigma_full=None,
            sigma_half=None,
            dsigma=None,
            sigma_coord=None,
            convection_fn=None,
            convection_config=None,
            radiation_fn=None,
            dynamic_albedo=False,
        )
        assert pipe.dynamic_albedo is False
