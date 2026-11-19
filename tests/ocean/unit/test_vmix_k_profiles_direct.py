"""Direct unit tests for the implicit-solver K/A profile composer
``ocean/physics/vertical_mixing/k_profiles.compute_vertical_K_profiles``.

The momentum-viscosity composition (``constant`` + ``enhanced_diffusion`` and
the KPP-suppression guard) is already pinned in
``test_enhanced_diffusion_momentum.py``; the TKE / CATKE prognostic paths are in
``test_tke_integration.py`` / ``test_catke_integration.py``.  This file fills the
remaining leaf branches that no other test exercises directly:

  * the ``vertical_mixing.scheme == "none"`` short-circuit — the output is
    exactly the supplied background floors, at the interior-interface shape;
  * the ``constant`` scheme — K/A equal the configured constants (plus floor);
  * background floors are additive.

The KPP / TKE / CATKE composition paths need cell-centred velocity input and
are exercised through the model-level factories in ``test_mpas_physics.py`` /
``test_tke_integration.py`` / the enhanced-diffusion suite; here we pin the two
analytic branches (``none`` / ``constant``) whose output is exactly predictable.

All on a small lat-lon C-grid rest state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    ConstantVerticalMixingConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.k_profiles import (
    compute_vertical_K_profiles,
)
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def grid_z_state():
    grid = create_latlon_grid(n_lat=8, n_lon=12)
    z = create_ocean_z_star(n_levels=5, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
    return grid, z, state


def _base(**over):
    return dict(
        lateral_mixing=type(OceanPhysicsConfig().lateral_mixing)(scheme="none"),
        surface_forcing=type(OceanPhysicsConfig().surface_forcing)(scheme="none"),
        shortwave_penetration=None,
        convection=OceanConvectionConfig(scheme="none"),
        **over,
    )


class TestNoneScheme:
    def test_returns_background_floors_at_interface_shape(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="none"), **_base())
        K, A = compute_vertical_K_profiles(
            state, z, None, cfg, A_v_background=1e-4, K_v_background=2e-5)
        nlev = state.T.data.shape[-1]
        assert K.shape == state.T.data.shape[:-1] + (nlev - 1,)
        # With no scheme + no convection, output IS the supplied floors.
        assert jnp.allclose(K, 2e-5)
        assert jnp.allclose(A, 1e-4)

    def test_zero_floors_give_zero(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="none"), **_base())
        K, A = compute_vertical_K_profiles(state, z, None, cfg)
        assert jnp.allclose(K, 0.0)
        assert jnp.allclose(A, 0.0)


class TestConstantScheme:
    def test_constant_values(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(A_v=3e-3, K_v=7e-4),
            ),
            **_base())
        K, A = compute_vertical_K_profiles(state, z, None, cfg)
        assert jnp.allclose(K, 7e-4)
        assert jnp.allclose(A, 3e-3)
        assert bool(jnp.all(jnp.isfinite(K)))

    def test_background_floor_is_additive(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(A_v=1e-3, K_v=1e-4),
            ),
            **_base())
        K0, A0 = compute_vertical_K_profiles(state, z, None, cfg)
        K1, A1 = compute_vertical_K_profiles(
            state, z, None, cfg, A_v_background=5e-4, K_v_background=2e-4)
        assert jnp.allclose(K1 - K0, 2e-4)
        assert jnp.allclose(A1 - A0, 5e-4)


def test_tke_bundle_ddm_operands_change_literal_composition(
        grid_z_state, monkeypatch):
    """Run TKEEntryN2Bundle.ddm_operands through its k_profiles consumer."""
    from legoesm.ocean.physics.vertical_mixing import double_diffusion
    from legoesm.ocean.physics.vertical_mixing.double_diffusion import (
        DoubleDiffusionConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.tke import TKEEntryN2Bundle
    from legoesm.ocean.vertical import create_partial_cell_coordinate

    _, base_z, state = grid_z_state
    z_coord = create_partial_cell_coordinate(base_z, state.H_bathy.data)
    vmix = VerticalMixingConfig(
        scheme="constant",
        constant=ConstantVerticalMixingConfig(K_v=1.0e-5, A_v=2.0e-5),
        ddm=DoubleDiffusionConfig(enabled=True, form="nemo_literal"))
    cfg = OceanPhysicsConfig(vertical_mixing=vmix, **_base())
    shape = state.T.data.shape[:-1] + (state.T.data.shape[-1] - 1,)
    seen = []

    def fake_ddm(avt, avm, zdt, zds, rn2, wet, _cfg):
        seen.append((np.asarray(zdt), np.asarray(zds)))
        delta = jnp.mean(zdt - zds)
        return avt + delta, avt + 2.0 * delta, avm + 3.0 * delta

    monkeypatch.setattr(double_diffusion, "nemo_zdf_ddm", fake_ddm)

    def selected(zdt_value):
        bundle = TKEEntryN2Bundle(
            rn2=jnp.ones(shape), rn2b=jnp.ones(shape),
            gdepw_Kmm=jnp.ones(shape), e3w_Kmm=jnp.ones(shape),
            e3t_Kmm=jnp.ones(state.T.data.shape),
            ddm_operands=(jnp.full(shape, zdt_value), jnp.ones(shape)))
        return compute_vertical_K_profiles(
            state, z_coord, None, cfg, tke_n2_bundle=bundle,
            return_salt_K=True)

    on = selected(3.0)
    off = selected(1.0)
    assert len(seen) == 2
    assert float(on[0][0, 0, 0]) == pytest.approx(2.00001)
    assert float(on[2][0, 0, 0]) == pytest.approx(4.00001)
    assert float(off[0][0, 0, 0]) == pytest.approx(1.0e-5)
    assert not np.array_equal(np.asarray(on[0]), np.asarray(off[0]))
    from types import SimpleNamespace
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

    z_n2 = create_partial_cell_coordinate(
        base_z._replace(
            nemo_gdept_0=jnp.broadcast_to(
                jnp.abs(base_z.z_full_ref), state.T.data.shape),
            nemo_gdepw_0=jnp.broadcast_to(
                jnp.abs(base_z.z_half_ref[:-1]), state.T.data.shape),
            nemo_e3t_0=jnp.broadcast_to(base_z.dz_ref, state.T.data.shape),
            nemo_e3w_0=jnp.broadcast_to(base_z.dz_ref, state.T.data.shape),
            nemo_e3w_mesh_reference=True),
        state.H_bathy.data)
    n2_bundle = LatLonCGridOceanModel._tke_step_entry_n2_bundle(
        SimpleNamespace(
            _nemo_ws_test_hooks=SimpleNamespace(
                bn2_tracer_override=None, bn2_intermediate="",
                bn2_alpha_beta_override=None),
            _n2_nemo_before_tracers=lambda *a, **k: None),
        state, z_coord=z_n2,
        config=SimpleNamespace(
            physics=cfg._replace(vertical_mixing=vmix._replace(
                scheme="tke",
                tke=TKEConfig(
                    tke_n2_evaluation_stage="step_entry",
                    n2_mode="nemo_bn2", n2_eos_form="eos80"))),
            gm_redi=None, eos_nemo_seos=None,
            constants=SimpleNamespace(g=9.80665)))
    assert n2_bundle.ddm_operands is not None and len(n2_bundle.ddm_operands) == 2


def test_tke_bundle_iwm_stretch_changes_literal_composition(
        grid_z_state, monkeypatch):
    """Run TKEEntryN2Bundle.iwm_stretch through its k_profiles consumer."""
    from legoesm.ocean.physics.vertical_mixing import internal_wave_mixing
    from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
        IWMConfig, IWMForcing,
    )
    from legoesm.ocean.physics.vertical_mixing.tke import TKEEntryN2Bundle
    from legoesm.ocean.vertical import create_partial_cell_coordinate

    _, base_z, state = grid_z_state
    raw_e3w = jnp.broadcast_to(
        base_z.dz_ref, state.T.data.shape)
    base_z = base_z._replace(
        t_depth_ref=jnp.abs(base_z.z_full_ref), nemo_e3w_0=raw_e3w)
    z_coord = create_partial_cell_coordinate(base_z, state.H_bathy.data)
    vmix = VerticalMixingConfig(
        scheme="constant",
        constant=ConstantVerticalMixingConfig(K_v=1.0e-5, A_v=2.0e-5),
        iwm=IWMConfig(enabled=True, form="nemo_literal", tsdiff=False))
    cfg = OceanPhysicsConfig(vertical_mixing=vmix, **_base())
    horizontal = state.eta.data.shape
    shape = state.T.data.shape[:-1] + (state.T.data.shape[-1] - 1,)
    forcing = IWMForcing(
        ebot=jnp.ones(horizontal), ecri=jnp.ones(horizontal),
        ensq=jnp.ones(horizontal), esho=jnp.ones(horizontal),
        hbot=jnp.ones(horizontal), hcri_inv=jnp.ones(horizontal))
    seen = []

    def fake_iwm(avt, avs, avm, _forcing, rn2, stretch, *_args, **_kwargs):
        seen.append(np.asarray(stretch))
        delta = jnp.mean(stretch)
        return avt + delta, avs + 2.0 * delta, avm + 3.0 * delta

    monkeypatch.setattr(internal_wave_mixing, "nemo_zdf_iwm", fake_iwm)

    def selected(stretch_value):
        bundle = TKEEntryN2Bundle(
            rn2=jnp.ones(shape), rn2b=jnp.ones(shape),
            gdepw_Kmm=jnp.ones(shape), e3w_Kmm=jnp.ones(shape),
            e3t_Kmm=jnp.ones(state.T.data.shape),
            iwm_stretch=jnp.full(horizontal, stretch_value))
        return compute_vertical_K_profiles(
            state, z_coord, None, cfg, tke_n2_bundle=bundle,
            iwm_fields=forcing, return_salt_K=True)

    on = selected(1.25)
    off = selected(1.0)
    assert len(seen) == 2
    assert float(on[0][0, 0, 0]) == pytest.approx(1.25001)
    assert float(on[2][0, 0, 0]) == pytest.approx(2.50001)
    assert float(off[0][0, 0, 0]) == pytest.approx(1.00001)
    assert not np.array_equal(np.asarray(on[0]), np.asarray(off[0]))
    from types import SimpleNamespace
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

    z_n2 = create_partial_cell_coordinate(
        base_z._replace(
            nemo_gdept_0=jnp.broadcast_to(
                jnp.abs(base_z.z_full_ref), state.T.data.shape),
            nemo_gdepw_0=jnp.broadcast_to(
                jnp.abs(base_z.z_half_ref[:-1]), state.T.data.shape),
            nemo_e3t_0=jnp.broadcast_to(base_z.dz_ref, state.T.data.shape),
            nemo_e3w_0=jnp.broadcast_to(base_z.dz_ref, state.T.data.shape),
            nemo_e3w_mesh_reference=True),
        state.H_bathy.data)
    n2_bundle = LatLonCGridOceanModel._tke_step_entry_n2_bundle(
        SimpleNamespace(
            _nemo_ws_test_hooks=SimpleNamespace(
                bn2_tracer_override=None, bn2_intermediate="",
                bn2_alpha_beta_override=None),
            _n2_nemo_before_tracers=lambda *a, **k: None),
        state, z_coord=z_n2,
        config=SimpleNamespace(
            physics=cfg._replace(vertical_mixing=vmix._replace(
                scheme="tke",
                tke=TKEConfig(
                    tke_n2_evaluation_stage="step_entry",
                    n2_mode="nemo_bn2", n2_eos_form="eos80"))),
            gm_redi=None, eos_nemo_seos=None,
            constants=SimpleNamespace(g=9.80665)))
    assert n2_bundle.iwm_stretch is not None


class TestBackgroundCompositionMode:
    """T23 (Phase-2 #1317): VerticalMixingConfig.vmix_background_mode.

    NEMO's own backgrounds compose by MAX (zdftke avm=max(...,avm0);
    zdfevd REPLACES where unstable) — never ADDED on top of another
    background. Default "additive" reproduces the prior (pre-T23)
    behaviour exactly; "nemo_max_floor" is the fix."""

    def test_default_is_additive_byte_identical(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(A_v=1e-3, K_v=1e-4),
            ),
            **_base())
        K_default, A_default = compute_vertical_K_profiles(
            state, z, None, cfg, A_v_background=5e-4, K_v_background=2e-4)
        cfg_explicit = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(A_v=1e-3, K_v=1e-4),
                vmix_background_mode="additive",
            ),
            **_base())
        K_explicit, A_explicit = compute_vertical_K_profiles(
            state, z, None, cfg_explicit,
            A_v_background=5e-4, K_v_background=2e-4)
        assert jnp.array_equal(K_default, K_explicit)
        assert jnp.array_equal(A_default, A_explicit)

    def test_nemo_max_floor_never_adds(self, grid_z_state):
        """MAX-floor composition: with the closure's own K_v (1e-4) BELOW
        the model-level floor (2e-4), the total is the floor ALONE (not
        1e-4+2e-4=3e-4 as the additive default would give)."""
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(A_v=1e-3, K_v=1e-4),
                vmix_background_mode="nemo_max_floor",
            ),
            **_base())
        K, A = compute_vertical_K_profiles(
            state, z, None, cfg, A_v_background=2e-4, K_v_background=2e-4)
        assert jnp.allclose(K, 2e-4)     # max(2e-4, 1e-4), NOT 3e-4
        assert jnp.allclose(A, 1e-3)     # max(2e-4, 1e-3), NOT 1.2e-3

    def test_nemo_max_floor_never_lowers_below_nemo_floor(self, grid_z_state):
        """SELF-REVIEW sign check: the MAX composition must never produce a
        total BELOW either input floor (a genuine floor, not a leak)."""
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(A_v=5e-4, K_v=3e-4),
                vmix_background_mode="nemo_max_floor",
            ),
            **_base())
        K, A = compute_vertical_K_profiles(
            state, z, None, cfg, A_v_background=1e-4, K_v_background=1e-4)
        assert bool(jnp.all(K >= 3e-4 - 1e-15))
        assert bool(jnp.all(K >= 1e-4 - 1e-15))
        assert bool(jnp.all(A >= 5e-4 - 1e-15))
        assert bool(jnp.all(A >= 1e-4 - 1e-15))

    def test_unknown_background_mode_raises(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant", vmix_background_mode="bogus",
            ),
            **_base())
        with pytest.raises(ValueError, match="vmix_background_mode"):
            compute_vertical_K_profiles(state, z, None, cfg)

    def test_nemo_max_floor_composes_with_convection_stable_branch(
        self, grid_z_state,
    ):
        """The EVD "stable branch" background (K_bg, the non-firing zdfevd
        value everywhere N2>=threshold — this rest state is stably
        stratified) must fold into the SAME max as the model-level floor
        under nemo_max_floor, NOT add a second time on top."""
        from legoesm.ocean.physics.convection.config import (
            EnhancedDiffusionConfig,
        )
        _, z, state = grid_z_state
        base_kwargs = dict(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(A_v=1e-3, K_v=1e-4),
            ),
            lateral_mixing=type(OceanPhysicsConfig().lateral_mixing)(
                scheme="none"),
            surface_forcing=type(OceanPhysicsConfig().surface_forcing)(
                scheme="none"),
            shortwave_penetration=None,
        )
        conv = OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(K_bg=2e-4, K_conv=1.0),
        )
        cfg_add = OceanPhysicsConfig(convection=conv, **base_kwargs)
        cfg_max = OceanPhysicsConfig(
            convection=conv,
            vertical_mixing=base_kwargs["vertical_mixing"]._replace(
                vmix_background_mode="nemo_max_floor"),
            **{k: v for k, v in base_kwargs.items()
               if k != "vertical_mixing"},
        )
        K_add, _ = compute_vertical_K_profiles(
            state, z, None, cfg_add, K_v_background=2e-4)
        K_max, _ = compute_vertical_K_profiles(
            state, z, None, cfg_max, K_v_background=2e-4)
        # additive: closure K_v (1e-4) + model floor (2e-4) + EVD stable
        # background (2e-4) = 5e-4 (stably stratified rest state -> EVD
        # never fires the K_conv branch).
        assert jnp.allclose(K_add, 5e-4)
        # nemo_max_floor: max(1e-4, 2e-4, 2e-4) = 2e-4 — nothing added.
        assert jnp.allclose(K_max, 2e-4)


class TestUnknownSchemeRaises:
    """Finding #3: an unknown vertical_mixing scheme must RAISE, not silently
    fall through to a zero K_v/A_v ('be defensive') return that disables vertical
    mixing on a typo (dispatch hardening; CLAUDE.md 'Dispatch')."""

    def test_unknown_scheme_raises_value_error(self, grid_z_state):
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="kpp_typo"), **_base())
        with pytest.raises(ValueError, match="unknown vertical_mixing.scheme"):
            compute_vertical_K_profiles(state, z, None, cfg)

    def test_none_scheme_still_returns_zeros(self, grid_z_state):
        """The explicit 'none' gate must keep returning background floors (here
        zero) — the raise must not catch the legitimate 'none' value."""
        _, z, state = grid_z_state
        cfg = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="none"), **_base())
        K, A = compute_vertical_K_profiles(state, z, None, cfg)
        assert jnp.allclose(K, 0.0)
        assert jnp.allclose(A, 0.0)


class TestSchemeReachability:
    """Reachability audit (execution half): every member of the single
    ``VALID_VERTICAL_MIXING_SCHEMES`` source must be DISPATCHABLE -- not a
    phantom that is selectable (reaches the runtime config, pinned in
    ``test_config_footguns.test_vertical_mixing_reachable_via_the_yaml_ocean_key``)
    yet falls straight through to the unknown-scheme ``raise`` at runtime.

    Behavioural, not AST: we actually invoke each dispatcher and assert the
    branch is ENTERED. A scheme that needs more inputs than the minimal call
    supplies (``catke`` wants ``dt_tke``; ``kpp``/``richardson`` want cell-centre
    velocities) still raises -- but NOT the unknown-scheme error -- which proves
    its branch was reached. Only a genuinely absent branch produces the
    unknown-scheme ``ValueError``.  Both the implicit K-profile dispatcher and
    the explicit-composition factory are covered, since a scheme can be
    unreachable on either path (codex finding #2)."""

    _UNKNOWN = "unknown vertical_mixing.scheme"

    def test_implicit_kprofile_dispatch_reaches_every_canonical_scheme(
        self, grid_z_state
    ):
        from legoesm.ocean.physics.vertical_mixing.config import (
            VALID_VERTICAL_MIXING_SCHEMES,
        )

        _, z, state = grid_z_state
        for scheme in sorted(VALID_VERTICAL_MIXING_SCHEMES):
            cfg = OceanPhysicsConfig(
                vertical_mixing=VerticalMixingConfig(scheme=scheme), **_base())
            try:
                compute_vertical_K_profiles(state, z, None, cfg)
            except Exception as exc:  # noqa: BLE001 - any non-unknown error = branch reached
                assert self._UNKNOWN not in str(exc), (
                    f"scheme {scheme!r} hit the unknown-scheme raise in "
                    f"compute_vertical_K_profiles -- it is selectable but has no "
                    f"dispatch branch (phantom): {exc}"
                )

    def test_explicit_factory_dispatch_reaches_every_canonical_scheme(self):
        from legoesm.ocean.physics.vertical_mixing.config import (
            VALID_VERTICAL_MIXING_SCHEMES,
        )
        from legoesm.ocean.physics.vertical_mixing.integration import (
            make_vertical_mixing_physics,
        )

        for scheme in sorted(VALID_VERTICAL_MIXING_SCHEMES):
            cfg = VerticalMixingConfig(scheme=scheme)
            # apply_diffusion=False (the implicit/host-model route) avoids the
            # explicit-only iwm/ddm NotImplementedError guards.
            fn = make_vertical_mixing_physics(cfg, apply_diffusion=False)
            assert callable(fn), (
                f"scheme {scheme!r} did not yield a callable from the explicit "
                f"vertical-mixing factory"
            )

    def test_explicit_factory_rejects_unknown_scheme(self):
        from legoesm.ocean.physics.vertical_mixing.integration import (
            make_vertical_mixing_physics,
        )

        with pytest.raises(ValueError, match="unknown vertical_mixing.scheme"):
            make_vertical_mixing_physics(
                VerticalMixingConfig(scheme="kpp_typo"), apply_diffusion=False)
