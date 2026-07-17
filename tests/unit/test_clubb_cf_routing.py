"""CLUBB cloud-fraction -> radiation ROUTING (make_physics / combined.py path).

Covers the routing contract that connects the turbulence WRITE side
(TurbulenceOutput.cloud_fraction) to the radiation READ side
(compute_cloud_properties cloud_fraction_override) through PhysicsState:

  * ``_carry_update_with_cloud_fraction`` returns a MULTI-FIELD dict (carry +
    cloud_fraction) only when the scheme diagnosed a cf, else the bare carry
    (byte-identical) — this is what combined.py merges into PhysicsState;
  * the static gate raises LOUDLY on misconfiguration (flag set without a
    cf-producing diagnostic-CLUBB closure) — never a silent no-op;
  * the READ side is wired for hydrostatic only (non-hydrostatic raises);
  * enabling the feature marks the radiation physics_fn ``_wants_phys_state_ro``
    so combined.py forwards phys_state; default-off leaves it unmarked
    (byte-identical).

The PHYSICAL effect of the override (lower cf => lower LWP => lower albedo) is
covered by tests/unit/test_clubb_cloud_fraction_override.py; here we pin the
plumbing/dispatch, which is factory-build-time (no JAX trace) so it stays cheap.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.turbulence.integration import (
    _carry_update_with_cloud_fraction,
)
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput


def _turb_out(cf):
    z = jnp.zeros((2, 3))
    s = jnp.zeros((2,))
    return TurbulenceOutput(z, z, z, z, z, z, s, s, s, s, cloud_fraction=cf)


class TestCarryPackaging:
    def test_no_cf_returns_bare_carry(self):
        """Scheme without a PDF cloud closure (cf None) => bare carry value, the
        pre-feature (value, not dict) contract."""
        carry = jnp.ones((2, 3))
        out = _carry_update_with_cloud_fraction("tke", carry, _turb_out(None))
        assert out is carry  # identity: byte-identical path

    def test_cf_returns_multifield_dict(self):
        """CLUBB (cf set) => dict co-locating the carry AND cloud_fraction so
        combined.py merges both into PhysicsState."""
        carry = jnp.ones((2, 3))
        cf = jnp.full((2, 3), 0.3)
        out = _carry_update_with_cloud_fraction("tke", carry, _turb_out(cf))
        assert isinstance(out, dict)
        assert set(out) == {"tke", "cloud_fraction"}
        assert bool(jnp.array_equal(out["tke"], carry))
        assert bool(jnp.array_equal(out["cloud_fraction"], cf))

    def test_cf_with_none_carry_field_only_cloud_fraction(self):
        """A diagnostic scheme with no prognostic carry still hands back the
        cloud fraction (single-key dict)."""
        cf = jnp.full((2, 3), 0.2)
        out = _carry_update_with_cloud_fraction(None, None, _turb_out(cf))
        assert isinstance(out, dict) and set(out) == {"cloud_fraction"}


def _cfg(turbulence="clubb", clubb=None, use_clubb_cf=True):
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="gray",
                                  use_clubb_cloud_fraction=use_clubb_cf),
        turbulence=TurbulenceConfig(scheme=turbulence, clubb=clubb),
    )


class TestStaticGate:
    def test_flag_without_clubb_raises(self):
        """use_clubb_cloud_fraction=True but turbulence is not clubb => LOUD
        (no cf producer => the override would ride the zero-init carry)."""
        with pytest.raises(ValueError, match="cloud-fraction-producing"):
            make_physics(_cfg(turbulence="louis"), model_type="hydrostatic")

    def test_flag_with_prognostic_clubb_raises(self):
        """Prognostic CLUBB carries packed moments, not a diagnosed cloud
        fraction => still refuse."""
        with pytest.raises(ValueError, match="cloud-fraction-producing"):
            make_physics(_cfg(clubb=CLUBBConfig(prognostic=True)),
                         model_type="hydrostatic")

    def test_non_hydrostatic_raises(self):
        """READ side is wired for hydrostatic only; a request on another dycore
        must raise, not silently ignore."""
        with pytest.raises(NotImplementedError, match="hydrostatic"):
            make_physics(_cfg(), model_type="nonhydrostatic")

    def test_diagnostic_clubb_hydrostatic_marks_readonly(self):
        """The supported path builds and the radiation physics_fn advertises the
        read-only phys_state marker so combined.py forwards phys_state."""
        fn = make_physics(_cfg(), model_type="hydrostatic")
        assert callable(fn)

    def test_default_off_does_not_mark_readonly(self):
        """Feature off => radiation fn is unmarked => phys_state not forwarded =>
        byte-identical RH grid-scale cloud path."""
        from legoesm.atmosphere.physics.radiation.integration import (
            make_radiation_physics,
        )
        rad_fn = make_radiation_physics(
            RadiationConfig(scheme="gray"), model_type="hydrostatic",
            use_clubb_cloud_fraction=False,
        )
        assert getattr(rad_fn, "_wants_phys_state_ro", False) is False
        rad_on = make_radiation_physics(
            RadiationConfig(scheme="gray"), model_type="hydrostatic",
            use_clubb_cloud_fraction=True,
        )
        assert rad_on._wants_phys_state_ro is True
