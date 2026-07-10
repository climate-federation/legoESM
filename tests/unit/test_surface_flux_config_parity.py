"""Cross-backend physics-config parity (#870 Phase 1).

The experiment-level surface bulk-flux settings and the tuned cloud scalars
must reach EVERY dycore backend through the shared builders:

- ``physics_pipeline.apply_surface_flux_config`` (called inside
  ``turbulence_config_for``) — so the MPAS/spectral standalone paths get the
  same COARE3/gustiness/stability surface layer the FV pipeline gets.
- ``model_driver._standalone_cloud_config`` — so the MPAS/spectral radiation
  clouds get the tuned ``cloud_q_c_diagnostic`` / Xu-Randall scalars.

Before #870 the injection lived only in the FV ``_resolve_turbulence`` and
the standalone paths silently ran scheme defaults.
"""

from __future__ import annotations

from types import SimpleNamespace

from legoesm.driver.model_driver import _standalone_cloud_config
from legoesm.driver.physics_pipeline import (
    apply_surface_flux_config,
    turbulence_config_for,
)
from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn


def _amip_like_config(**overrides):
    """Minimal config carrying the production surface-flux fields."""
    base = dict(
        turbulence="louis",
        turbulence_override=None,
        surface_bulk_scheme="coare3",
        surface_gustiness_zi=300.0,
        surface_thermo_convention="legoesm",
        surface_stability_scheme="beljaars_holtslag1991",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_surface_fields_injected_into_active_scheme():
    tc = turbulence_config_for(_amip_like_config())
    surf = tc.louis.surface
    assert surf.bulk_scheme == "coare3"
    assert surf.gustiness_w_zi == 300.0
    assert surf.stability_scheme == "beljaars_holtslag1991"


def test_default_config_returns_same_object():
    """Byte-identical guard: no surface fields set => tc passes through
    UNCHANGED (the override identity contract elsewhere relies on this)."""
    cfg = _amip_like_config(
        surface_bulk_scheme="constant",
        surface_gustiness_zi=None,
        surface_stability_scheme="dyer1974",
    )
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    tc = TurbulenceConfig(scheme="louis")
    assert apply_surface_flux_config(tc, cfg) is tc


def test_fv_resolver_parity_via_get_turbulence_fn():
    """What the FV ``_resolve_turbulence`` returns (the scheme sub-config)
    carries the patched surface — the injection moved INTO the shared
    ``turbulence_config_for``, so the sub-config must arrive pre-patched."""
    tc = turbulence_config_for(_amip_like_config())
    _name, _fn, sub = get_turbulence_fn(tc)
    assert sub.surface.bulk_scheme == "coare3"
    assert sub.surface.gustiness_w_zi == 300.0


def test_scheme_none_is_safe():
    tc = turbulence_config_for(_amip_like_config(turbulence="none"))
    assert tc.scheme == "none"


def test_standalone_cloud_config_carries_tuned_scalars():
    cfg = SimpleNamespace(
        cloud_rh_crit=None,
        cloud_q_c_diagnostic=3.0e-4,
        cloud_conv_cloud_max=None,
        cloud_conv_cloud_condensate=None,
        cloud_p_xr=0.6,
        cloud_alpha_xr=None,
        convective_cloud=True,  # must NOT propagate (no conv_precip threaded)
    )
    cc = _standalone_cloud_config(cfg, "sundqvist")
    assert cc.q_c_diagnostic == 3.0e-4
    assert cc.p_xr == 0.6
    # standalone radiation does not thread conv_precip: forced OFF by design
    assert cc.convective_cloud is False


def test_standalone_cloud_config_none_scheme():
    assert _standalone_cloud_config(SimpleNamespace(), "none") is None
