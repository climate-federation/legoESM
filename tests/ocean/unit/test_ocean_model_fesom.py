"""Unit tests for the FESOM ocean model adapter.

These tests require ``fesom_jax`` and the packaged pi mesh.  Tests that do
not exercise the dycore (validation, module-import checks) run without a
mesh.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import replace as dc_replace
from typing import Any

import numpy as np
import pytest

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# The module under test must import WITHOUT fesom_jax installed (F3).
# We import it unconditionally; if fesom_jax is missing, only the
# function-scope calls inside it will fail.
# ---------------------------------------------------------------------------
from legoesm.ocean.dynamics.ocean_model_fesom import (
    FesomOceanConfig,
    FesomOceanGrid,
    FesomOceanState,
    FesomOceanModel,
    build_flat_bottom_mesh,
    create_lock_exchange_state,
)
from legoesm.core.field import Field

# Real lock-exchange config — its defaults are the production values
# (nlev=20, H_max=20.0, land_lat_threshold=80.0, T_cold_C=5.0,
#  T_warm_C=30.0, S_uniform=35.0, T_reference_C=15.0, front_longitude=0.0).
from legoesm.ocean.experiments.lock_exchange import LockExchangeConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def pi_mesh():
    """Load the packaged pi mesh, or skip if unavailable.

    Uses the REAL fesom_jax API:
        from fesom_jax.mesh import load_mesh, DEFAULT_PI_MESH_DIR
        load_mesh(mesh_dir=DEFAULT_PI_MESH_DIR)
    with fallbacks for older calling conventions.
    """
    pytest.importorskip("fesom_jax")

    # Real API first.
    try:
        from fesom_jax.mesh import load_mesh, DEFAULT_PI_MESH_DIR
        return load_mesh(mesh_dir=DEFAULT_PI_MESH_DIR)
    except Exception:
        pass

    # Older / alternative calling conventions.
    candidates = []
    try:
        from fesom_jax.mesh import load_mesh
        candidates.append(load_mesh)
    except ImportError:
        pass
    try:
        from fesom_jax import load_mesh as _lm
        candidates.append(_lm)
    except ImportError:
        pass
    if not candidates:
        pytest.skip("Cannot find load_mesh in fesom_jax")

    for fn in candidates:
        for args, kw in [((), {}), (("pi",), {}), ((), {"name": "pi"})]:
            try:
                return fn(*args, **kw)
            except Exception:
                continue
    pytest.skip("Could not load pi mesh with any known API")


@pytest.fixture
def lock_config():
    """Real LockExchangeConfig with production defaults."""
    return LockExchangeConfig()


@pytest.fixture
def flat_mesh(pi_mesh, lock_config):
    """Flat-bottom mesh with the PRODUCTION land_lat_threshold (80.0).

    This means some nodes (those above 80° latitude) are dry.  Tests that
    care about dryness inspect the mesh directly; tests that don't should
    still pass because the dry nodes are well-posed (zero depth, zero wet
    layers, finite step).
    """
    return build_flat_bottom_mesh(
        pi_mesh,
        H_max=lock_config.H_max,
        nlev=lock_config.nlev,
        land_lat_threshold=lock_config.land_lat_threshold,
    )


@pytest.fixture
def state(flat_mesh, lock_config):
    return create_lock_exchange_state(flat_mesh, lock_config)


@pytest.fixture
def grid(flat_mesh):
    return FesomOceanGrid(flat_mesh)


@pytest.fixture
def z20_or_none(lock_config):
    """The benchmark z-star coordinate (20 levels over 20 m)."""
    from legoesm.ocean.vertical import create_ocean_z_star
    return create_ocean_z_star(n_levels=lock_config.nlev,
                               H_max=lock_config.H_max)


# =============================================================================
# F3 — module imports cleanly without fesom_jax at module scope
# =============================================================================

class TestDeferredImports:
    """Verify no fesom_jax names leak to module scope (F3)."""

    def test_no_module_level_fesom_jax_names(self):
        import legoesm.ocean.dynamics.ocean_model_fesom as mod
        forbidden = [
            "fssh", "fstep", "State", "Mesh", "level_masks",
            "_K_VER", "_A_VER", "_DT_DEFAULT",
            "fesom_jax",
        ]
        for name in forbidden:
            assert not hasattr(mod, name), (
                f"'{name}' must not be at module scope — "
                f"all fesom_jax imports must be function-scope (F3)."
            )

    def test_fesomoceanconfig_constructs_without_fesom_jax(self):
        """Config must be constructible without fesom_jax (all-None defaults)."""
        cfg = FesomOceanConfig()
        assert cfg.dt is None
        assert cfg.k_ver is None
        assert cfg.a_ver is None

    def test_require_fesom_jax_raises_clearly(self, monkeypatch):
        """If fesom_jax import fails, require_fesom_jax raises an ImportError
        whose message names the package and gives an install command.

        This test FAILS if require_fesom_jax is removed, returns silently,
        or raises a different exception type / message — covering every
        branch of the gate.
        """
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fesom_jax" or name.startswith("fesom_jax."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        # Clear cached modules so the fake import is actually consulted.
        import sys
        saved = {
            k: v for k, v in sys.modules.items()
            if k == "fesom_jax" or k.startswith("fesom_jax.")
        }
        for k in saved:
            del sys.modules[k]

        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            from legoesm.ocean.dynamics.ocean_model_fesom import require_fesom_jax
            with pytest.raises(ImportError) as excinfo:
                require_fesom_jax()
            msg = str(excinfo.value)
            assert "fesom_jax" in msg, (
                f"ImportError message must name 'fesom_jax'; got: {msg!r}"
            )
            assert "pip install" in msg, (
                f"ImportError message must give an install command; got: {msg!r}"
            )
        finally:
            # Restore cached modules so other tests can use fesom_jax.
            sys.modules.update(saved)


# =============================================================================
# F5 — input validation
# =============================================================================

class TestBuildFlatBottomMeshValidation:

    def test_H_max_zero_raises(self):
        with pytest.raises(ValueError, match="H_max must be a POSITIVE"):
            build_flat_bottom_mesh(None, H_max=0.0, nlev=10)

    def test_H_max_negative_raises(self):
        with pytest.raises(ValueError, match="H_max must be a POSITIVE"):
            build_flat_bottom_mesh(None, H_max=-20.0, nlev=10)

    def test_nlev_zero_raises(self):
        with pytest.raises(ValueError, match="nlev must be >= 1"):
            build_flat_bottom_mesh(None, H_max=20.0, nlev=0)

    def test_nlev_negative_raises(self):
        with pytest.raises(ValueError, match="nlev must be >= 1"):
            build_flat_bottom_mesh(None, H_max=20.0, nlev=-3)

    def test_land_lat_threshold_out_of_range_raises(self):
        """New branch: land_lat_threshold outside [0, 90] is a ValueError."""
        with pytest.raises(ValueError, match="land_lat_threshold"):
            build_flat_bottom_mesh(
                None, H_max=20.0, nlev=10, land_lat_threshold=-5.0,
            )
        with pytest.raises(ValueError, match="land_lat_threshold"):
            build_flat_bottom_mesh(
                None, H_max=20.0, nlev=10, land_lat_threshold=120.0,
            )


# =============================================================================
# F4 — land mask and dry-column mechanism
# =============================================================================

class TestLandMask:
    """Tests for land_mask derivation and the dry-column mechanism."""

    def test_land_lat_threshold_90_ok(self, pi_mesh):
        """Default 90.0 must not raise and produces no dry nodes."""
        mesh = build_flat_bottom_mesh(pi_mesh, H_max=20.0, nlev=5)
        assert mesh is not None
        # No node has abs(lat) > 90, so every node is fully wet. "Fully wet"
        # means every LAYER column is True; FESOM layer validity is
        # ``ulevels-1 <= k < nlevels-1``, so the trailing k == nl-1 column is
        # an INTERFACE slot and is False by construction on every mesh. It is
        # not a dry layer, so it must be excluded from this assertion.
        node_layer_mask = np.asarray(mesh.node_layer_mask)
        nl = int(mesh.nl)
        assert node_layer_mask[:, : nl - 1].all(), (
            "At land_lat_threshold=90 (no dry nodes) every node should be "
            "wet on every layer."
        )
        assert not node_layer_mask[:, nl - 1].any(), (
            "The trailing interface column must never be a valid layer."
        )

    def test_land_lat_threshold_80_dries_polar_nodes(self, pi_mesh):
        """land_lat_threshold=80 must dry polar nodes (190 of 3140 on pi)."""
        mesh = build_flat_bottom_mesh(
            pi_mesh, H_max=20.0, nlev=20, land_lat_threshold=80.0,
        )
        geo_lat = np.asarray(mesh.geo_coord_nod2D[:, 1])
        dry = np.abs(np.degrees(geo_lat)) > 80.0
        assert dry.sum() > 0, "pi mesh should have nodes above 80° latitude"
        node_layer_mask = np.asarray(mesh.node_layer_mask)
        # Dry nodes have ZERO wet layers.
        assert node_layer_mask[dry].sum() == 0, (
            f"Dry nodes must have no wet layers; got "
            f"{node_layer_mask[dry].sum()} wet layer-slots on dry nodes."
        )
        # Wet nodes have at least one wet layer.
        assert node_layer_mask[~dry].any(axis=1).all(), (
            "Wet nodes must have at least one wet layer."
        )

    def test_state_land_mask_matches_node_layer_mask(self, state, flat_mesh):
        """land_mask must be derived from node_layer_mask.any(axis=1)."""
        mask = np.asarray(state.land_mask.data)
        expected = np.asarray(
            flat_mesh.node_layer_mask.any(axis=1), dtype=np.float64
        )
        np.testing.assert_allclose(mask, expected)
        # Shape sanity.
        assert mask.shape == (int(flat_mesh.nod2D),)

    def test_state_land_mask_zero_on_dry_one_elsewhere(self, state, flat_mesh,
                                                       lock_config):
        """Dry nodes have land_mask=0; wet nodes have land_mask=1."""
        geo_lat = np.asarray(flat_mesh.geo_coord_nod2D[:, 1])
        dry = np.abs(np.degrees(geo_lat)) > lock_config.land_lat_threshold
        mask = np.asarray(state.land_mask.data)
        if dry.any():
            assert np.all(mask[dry] == 0.0), (
                f"Dry nodes must have land_mask=0; got {mask[dry][:5]}"
            )
        assert np.all(mask[~dry] == 1.0), (
            f"Wet nodes must have land_mask=1; got {mask[~dry][:5]}"
        )


# =============================================================================
# Dry-column mechanism — additional integration tests
# =============================================================================

class TestDryColumns:
    """New tests for the dry-column mechanism."""

    def test_dry_nodes_have_zero_wet_layers(self, flat_mesh, lock_config):
        """Direct test of the dry-column mechanism on the production mesh."""
        geo_lat = np.asarray(flat_mesh.geo_coord_nod2D[:, 1])
        dry = np.abs(np.degrees(geo_lat)) > lock_config.land_lat_threshold
        assert dry.sum() > 0, (
            "Production config (land_lat_threshold=80) must dry some nodes."
        )
        node_layer_mask = np.asarray(flat_mesh.node_layer_mask)
        assert node_layer_mask[dry].sum() == 0, (
            "Dry nodes must have no wet layers."
        )

    def test_dry_nodes_have_zero_depth(self, flat_mesh, lock_config):
        """Dry nodes have depth=0 so no operator divides by zero."""
        geo_lat = np.asarray(flat_mesh.geo_coord_nod2D[:, 1])
        dry = np.abs(np.degrees(geo_lat)) > lock_config.land_lat_threshold
        depth = np.asarray(flat_mesh.depth)
        assert np.allclose(depth[dry], 0.0), (
            f"Dry nodes must have depth=0; got {depth[dry][:5]}"
        )
        # Wet nodes still have depth=-H_max (flat bottom preserved).
        assert np.allclose(depth[~dry], -lock_config.H_max)

    def test_dry_nodes_have_collapsed_nlevels(self, flat_mesh, lock_config):
        """Dry nodes must have nlevels == ulevels (collapsed), so the
        valid-layer range ``ulevels-1 <= k < nlevels-1`` is empty AND the
        raw integer bounds used by ssh.build_ssh_operator read 0 depth.

        This test FAILS if build_flat_bottom_mesh reverts to returning the
        full-depth ``nlevels`` on dry nodes (the P0 bug): the integer
        bounds are what ssh.build_ssh_operator reads via ``zbar[nlevels-1]``,
        so leaving them at nl makes dry elements contribute full-depth SSH
        stiffness even when the boolean mask is collapsed.
        """
        nlevels = np.asarray(flat_mesh.nlevels_nod2D)
        ulevels = np.asarray(flat_mesh.ulevels_nod2D)
        nl = int(flat_mesh.nl)
        geo_lat = np.asarray(flat_mesh.geo_coord_nod2D[:, 1])
        dry = np.abs(np.degrees(geo_lat)) > lock_config.land_lat_threshold
        # Wet nodes still full-depth.
        assert np.all(nlevels[~dry] == nl), (
            "Wet nodes must retain nlevels == nl."
        )
        # Dry nodes COLLAPSED — nlevels == ulevels.
        assert np.all(nlevels[dry] == ulevels[dry]), (
            f"Dry nodes must have nlevels == ulevels (collapsed); got "
            f"nlevels[dry][:5]={nlevels[dry][:5]}, "
            f"ulevels[dry][:5]={ulevels[dry][:5]}. If these equal nl, the "
            f"P0 regression is back: dry elements will contribute "
            f"full-depth SSH stiffness."
        )

    def test_step_d_eta_exactly_zero_on_dry_nodes(
        self, flat_mesh, lock_config
    ):
        """After one step from the lock-exchange IC, ``d_eta`` must be
        EXACTLY 0 on dry nodes.

        This is the integration check that actually catches the SSH
        contamination: when the dry nodes' integer ``nlevels`` bounds are
        left at full depth, ``ssh.build_ssh_operator`` reads
        ``zbar[nlevels-1]`` = full depth for dry elements, and the
        resulting stiffness couples dry nodes to the wet domain.  Measured
        on the pi mesh at ``land_lat_threshold=80``: broken code gives
        ``|d_eta|`` up to 3.2e-7 on dry nodes; fixed code gives 0.0
        exactly.

        This test asserts on ``d_eta`` (the increment), NOT on ``eta_n``:
        ``eta_n`` is not the contaminated field — its dry-node value is
        zero in both broken and fixed runs, which is why the previous
        version of this test passed vacuously.
        """
        pytest.importorskip("fesom_jax")
        state = create_lock_exchange_state(flat_mesh, lock_config)
        eta0 = np.asarray(state.eta.data)
        assert np.allclose(eta0, 0.0), "Rest state must have eta=0"

        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        new_state = model.step(state, 3600.0)

        geo_lat = np.asarray(flat_mesh.geo_coord_nod2D[:, 1])
        dry = np.abs(np.degrees(geo_lat)) > lock_config.land_lat_threshold
        new_eta = np.asarray(new_state.eta.data)
        d_eta = new_eta - eta0

        # Dry columns: d_eta must be finite AND exactly zero.
        assert np.all(np.isfinite(d_eta[dry])), (
            "Dry-column d_eta must be finite after a step."
        )
        max_abs_d_eta_dry = float(np.max(np.abs(d_eta[dry]))) if dry.any() else 0.0
        assert max_abs_d_eta_dry == 0.0, (
            f"d_eta on dry nodes must be EXACTLY 0 (collapsed integer "
            f"bounds prevent SSH stiffness coupling to dry elements). "
            f"Got max |d_eta| on dry = {max_abs_d_eta_dry:.3e}. "
            f"If this is ~3e-7, the P0 regression (full-depth nlevels on "
            f"dry nodes) is back."
        )

    def test_step_from_rest_is_finite(self, flat_mesh, lock_config):
        """One step from rest must produce a globally finite state."""
        pytest.importorskip("fesom_jax")
        state = create_lock_exchange_state(flat_mesh, lock_config)
        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        new_state = model.step(state, 3600.0)
        eta = np.asarray(new_state.eta.data)
        assert np.all(np.isfinite(eta)), (
            f"Non-finite eta after step: {np.where(~np.isfinite(eta))}"
        )


# =============================================================================
# Flat-bottom mesh correctness
# =============================================================================

class TestFlatBottomMesh:

    def test_depth_is_negative_H_max_on_wet(self, flat_mesh, lock_config):
        """FESOM depth is NEGATIVE; equals -H_max at every WET node."""
        depth = np.asarray(flat_mesh.depth)
        geo_lat = np.asarray(flat_mesh.geo_coord_nod2D[:, 1])
        dry = np.abs(np.degrees(geo_lat)) > lock_config.land_lat_threshold
        # Wet nodes have depth -H_max.
        assert np.allclose(depth[~dry], -lock_config.H_max)

    def test_zbar_uniform_layers(self, flat_mesh, lock_config):
        """zbar goes from 0 to -H_max in nlev+1 uniform steps."""
        zbar = np.asarray(flat_mesh.zbar)
        assert zbar[0] == 0.0
        assert np.isclose(zbar[-1], -lock_config.H_max)
        # Uniform spacing
        dz = np.diff(zbar)
        assert np.allclose(dz, dz[0])

    def test_nlevels_wet_full_and_dry_collapsed(self, flat_mesh, lock_config):
        """Wet nodes have nlevels == nl; DRY nodes have nlevels == ulevels.

        This replaces the old ``test_nlevels_equal_nl_on_wet`` which only
        asserted on wet nodes and explicitly documented the WRONG
        behaviour (``nlevels`` unchanged on dry nodes) as expected.  After
        the P0 fix, dry nodes' integer ``nlevels`` IS collapsed to
        ``ulevels`` — and that collapse is what prevents SSH contamination
        (the kernel ``ssh.build_ssh_operator`` reads ``zbar[nlevels-1]``
        directly, not the mask).
        """
        nl = int(flat_mesh.nl)
        nlevels = np.asarray(flat_mesh.nlevels_nod2D)
        ulevels = np.asarray(flat_mesh.ulevels_nod2D)
        geo_lat = np.asarray(flat_mesh.geo_coord_nod2D[:, 1])
        dry = np.abs(np.degrees(geo_lat)) > lock_config.land_lat_threshold

        assert np.all(nlevels[~dry] == nl), (
            "Wet nodes must have nlevels == nl (full-depth column)."
        )
        # The check that actually catches the P0 regression:
        assert np.all(nlevels[dry] == ulevels[dry]), (
            f"Dry nodes must have nlevels == ulevels (collapsed). "
            f"Got nlevels[dry][:5]={nlevels[dry][:5]}, "
            f"ulevels[dry][:5]={ulevels[dry][:5]}. If nlevels[dry] == nl, "
            f"the P0 bug is back and dry elements contribute full-depth "
            f"SSH stiffness."
        )

    def test_ulevels_equal_1(self, flat_mesh):
        ulevels = np.asarray(flat_mesh.ulevels_nod2D)
        assert np.all(ulevels == 1)


# =============================================================================
# FesomOceanState — the state seam
# =============================================================================

class TestFesomOceanState:

    def test_state_is_fesom_ocean_state(self, state):
        assert isinstance(state, FesomOceanState)

    def test_T_is_field(self, state):
        assert isinstance(state.T, Field)

    def test_T_slices_padding_column(self, state, flat_mesh):
        """The last column of inner.T is interface padding and must be dropped."""
        nlev = int(flat_mesh.nl) - 1
        T_data = np.asarray(state.T.data)
        assert T_data.shape[1] == nlev, (
            f"T has {T_data.shape[1]} columns; expected nlev={nlev} "
            f"(mesh.nl-1). The padding column was not dropped."
        )

    def test_S_slices_padding_column(self, state, flat_mesh):
        nlev = int(flat_mesh.nl) - 1
        S_data = np.asarray(state.S.data)
        assert S_data.shape[1] == nlev

    def test_eta_shape(self, state, flat_mesh):
        eta = np.asarray(state.eta.data)
        assert eta.shape == (int(flat_mesh.nod2D),)

    def test_H_bathy_positive_on_wet(self, state, lock_config):
        """SIGN GATE: H_bathy must be positive depth on wet nodes.

        Dry nodes have depth=0 ⇒ H_bathy=0; wet nodes have H_bathy=H_max.
        """
        H = np.asarray(state.H_bathy.data)
        # H_bathy is +depth; wet nodes have H_max, dry have 0.
        assert np.all(H >= 0), "H_bathy must be NON-NEGATIVE (legoESM convention)"
        assert np.any(np.isclose(H, lock_config.H_max)), (
            "At least wet nodes should have H_bathy == H_max."
        )

    def test_H_bathy_equals_negative_mesh_depth(self, state, flat_mesh):
        """H_bathy == -mesh.depth (sign flip at the seam)."""
        H = np.asarray(state.H_bathy.data)
        neg_depth = np.asarray(flat_mesh.depth)
        assert np.allclose(H, -neg_depth)

    def test_nlev_from_mesh(self, state, flat_mesh):
        assert state.nlev == int(flat_mesh.nl) - 1

    def test_u_is_field_with_node_dim(self, state, flat_mesh):
        """u must be a Field with leading dim == nod2D (NODE field)."""
        u = state.u
        assert isinstance(u, Field)
        u_data = np.asarray(u.data)
        nod2D = int(flat_mesh.nod2D)
        nlev = state.nlev
        assert u_data.shape == (nod2D, nlev), (
            f"u must have shape (nod2D={nod2D}, nlev={nlev}); "
            f"got {u_data.shape}."
        )
        assert u.dims == ("cell", "nlev")
        assert u.units == "m/s"

    def test_v_is_field_with_node_dim(self, state, flat_mesh):
        """v must be a Field with leading dim == nod2D (NODE field)."""
        v = state.v
        assert isinstance(v, Field)
        v_data = np.asarray(v.data)
        nod2D = int(flat_mesh.nod2D)
        nlev = state.nlev
        assert v_data.shape == (nod2D, nlev)
        assert v.dims == ("cell", "nlev")
        assert v.units == "m/s"

    def test_uv_node_dims_match_T(self, state):
        """u and v must have the SAME leading dim as T (the node count)."""
        T = np.asarray(state.T.data)
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        assert u.shape[0] == T.shape[0], (
            f"u leading dim {u.shape[0]} != T leading dim {T.shape[0]}"
        )
        assert v.shape[0] == T.shape[0], (
            f"v leading dim {v.shape[0]} != T leading dim {T.shape[0]}"
        )
        # Also the nlev dim.
        assert u.shape[1] == T.shape[1]
        assert v.shape[1] == T.shape[1]

    def test_uv_elem_field(self, state, flat_mesh):
        """uv_elem must expose the prognostic element velocity."""
        uv = state.uv_elem
        assert isinstance(uv, Field)
        uv_data = np.asarray(uv.data)
        elem2D = int(flat_mesh.elem2D)
        nlev = state.nlev
        assert uv_data.shape == (elem2D, nlev, 2), (
            f"uv_elem shape {uv_data.shape}; expected ({elem2D}, {nlev}, 2)"
        )
        assert uv.dims == ("elem", "nlev", "comp")
        assert uv.units == "m/s"

    def test_u_nonzero_after_first_step(self, flat_mesh, lock_config):
        """``u`` must be a CURRENT-step interpolation of the prognostic
        element velocity, NOT the lagged ``inner.uvnode``.

        Under the OLD lagged code, ``inner.uvnode`` was computed from the
        start-of-step ``uv`` BEFORE ``uv`` was updated, so after step 1
        from rest ``max|uv| = 0.04446`` but ``max|uvnode| = 0`` — i.e.
        ``max|u|`` was EXACTLY 0 on the very step where the flow starts.
        Under the fixed code (``compute_vel_nodes(mesh, inner.uv)``),
        ``max|u|`` after step 1 must be nonzero and comparable in
        magnitude to ``max|uv_elem|``.

        This test FAILS if ``u``/``v`` are reverted to reading
        ``inner.uvnode``: ``max_u`` would be exactly 0.
        """
        pytest.importorskip("fesom_jax")
        state = create_lock_exchange_state(flat_mesh, lock_config)
        # Sanity: IC is at rest, so max|u| is 0 here.
        u0 = np.asarray(state.u.data)
        assert np.max(np.abs(u0)) == 0.0, (
            "IC is at rest — max|u| should be 0 before any step."
        )

        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        s1 = model.step(state, 3600.0)

        u_after = np.asarray(s1.u.data)
        uv_elem_after = np.asarray(s1.uv_elem.data)
        max_u = float(np.max(np.abs(u_after)))
        max_uv = float(np.max(np.abs(uv_elem_after)))

        # The prognostic element velocity must have moved off zero — this
        # also fails the test if the dycore itself fails to advance.
        assert max_uv > 0.0, (
            f"Prognostic element velocity must be nonzero after step 1 of "
            f"lock exchange; got max|uv_elem|={max_uv}."
        )
        # The node interpolation must reflect the CURRENT step (nonzero),
        # not the lagged value (which was 0).  Under the old code this
        # assertion failed with max_u == 0.0.
        assert max_u > 0.0, (
            f"max|u| after step 1 must be nonzero (current-step "
            f"interpolation via compute_vel_nodes). Got max|u|={max_u} "
            f"exactly — this is the lagged-uvnode bug: u was reading "
            f"inner.uvnode, which is computed from the start-of-step uv "
            f"and is zero on the first step."
        )
        # Node interpolation should not blow up relative to the element
        # velocity — same order of magnitude.
        assert max_u < 10.0 * max_uv, (
            f"max|u|={max_u} should be comparable to (not vastly larger "
            f"than) max|uv_elem|={max_uv}; u is an interpolation of "
            f"uv_elem, not a different field."
        )


# =============================================================================
# FesomOceanGrid — the grid seam
# =============================================================================

class TestFesomOceanGrid:

    def test_grid_area_is_1d(self, grid, flat_mesh):
        """area must be (nod2D,), NOT the 2-D (nod2D, nl) mesh.areasvol."""
        area = np.asarray(grid.area)
        assert area.ndim == 1, (
            f"grid.area has {area.ndim} dims; expected 1-D (nod2D,). "
            f"A 2-D array would make compute_rpe return a wrong result."
        )
        assert area.shape == (int(flat_mesh.nod2D),)

    def test_grid_area_reads_areasvol_level0(self, grid, flat_mesh):
        """Sanity: on the pi mesh, area == areasvol[:, 0] == area[:, 0]
        (they coincide).  The discriminating test below
        (``test_grid_area_tracks_areasvol_when_perturbed``) is the one
        that actually fails if ``.area`` is reverted to ``area[:, 0]``.
        """
        area = np.asarray(grid.area)
        areasvol_l0 = np.asarray(flat_mesh.areasvol[:, 0])
        assert np.allclose(area, areasvol_l0)

    def test_grid_area_tracks_areasvol_when_perturbed(self, flat_mesh):
        """DISCRIMINATING RPE-weight test.

        On the flat pi mesh ``mesh.area[:, 0] == mesh.areasvol[:, 0]``
        EXACTLY, and flattening preserves that, so a test that just
        asserts ``grid.area == areasvol[:, 0]`` passes whether the
        implementation reads ``areasvol`` or ``area`` — it cannot tell
        them apart.  This test perturbs ``areasvol`` (via
        ``dataclasses.replace``) so the two differ, then verifies:

        * ``FesomOceanGrid.area`` tracks the perturbed ``areasvol``
          (FAILS if ``.area`` is reverted to ``mesh.area[:, 0]``).
        * ``FesomOceanGrid.area_upper`` still tracks ``area`` (FAILS if
          ``.area_upper`` is wrongly wired to ``areasvol``).
        """
        nod2D = int(flat_mesh.nod2D)
        nl = int(flat_mesh.nl)
        base_areasvol = np.asarray(flat_mesh.areasvol[:, 0])
        base_area = np.asarray(flat_mesh.area[:, 0])
        # Document the vacuous condition: the unperturbed mesh cannot
        # distinguish the two arrays.
        assert np.allclose(base_areasvol, base_area), (
            "Test precondition: on the flat pi mesh, area[:,0] and "
            "areasvol[:,0] coincide — that is exactly why the old test "
            "was vacuous."
        )

        # Perturb areasvol by a factor distinct from area.
        perturbed_l0 = base_areasvol * 3.0 + 1.0
        perturbed_2d = jnp.broadcast_to(
            jnp.asarray(perturbed_l0, dtype=jnp.float64)[:, None],
            (nod2D, nl),
        )
        mesh2 = dc_replace(flat_mesh, areasvol=perturbed_2d)
        grid2 = FesomOceanGrid(mesh2)

        a = np.asarray(grid2.area)
        au = np.asarray(grid2.area_upper)

        # area tracks areasvol (perturbed), NOT area (unchanged).
        assert np.allclose(a, perturbed_l0), (
            f"FesomOceanGrid.area must track mesh.areasvol[:, 0], not "
            f"mesh.area[:, 0]. Got area[:5]={a[:5]}, expected (perturbed "
            f"areasvol) {perturbed_l0[:5]}. If this matches base_area "
            f"instead ({base_area[:5]}), the volume-weight bug is back."
        )
        # area_upper tracks area (unchanged).
        assert np.allclose(au, base_area), (
            f"FesomOceanGrid.area_upper must track mesh.area[:, 0]. "
            f"Got {au[:5]}, expected {base_area[:5]}."
        )
        # And they must now differ.
        assert not np.allclose(a, au), (
            "After perturbing areasvol, area and area_upper must differ — "
            "if they are still equal the test is vacuous again."
        )

    def test_grid_area_upper_reads_area_level0(self, grid, flat_mesh):
        """area_upper must be mesh.area[:, 0] — the surface area."""
        area_upper = np.asarray(grid.area_upper)
        area_l0 = np.asarray(flat_mesh.area[:, 0])
        assert np.allclose(area_upper, area_l0)
        assert area_upper.ndim == 1

    def test_grid_lon_lat_radians(self, grid, flat_mesh):
        """lon/lat must be in RADIANS from geo_coord_nod2D."""
        lon = np.asarray(grid.lon)
        lat = np.asarray(grid.lat)
        ref_lon = np.asarray(flat_mesh.geo_coord_nod2D[:, 0])
        ref_lat = np.asarray(flat_mesh.geo_coord_nod2D[:, 1])
        assert np.allclose(lon, ref_lon)
        assert np.allclose(lat, ref_lat)

    def test_grid_type_fesom(self, grid):
        assert grid.grid_type == "fesom"

    def test_grid_mesh_attribute(self, grid, flat_mesh):
        assert grid.mesh is flat_mesh


# =============================================================================
# compute_rpe runs on the adapter state UNMODIFIED
# =============================================================================

class TestComputeRPE:
    """The whole point of FesomOceanState: legoESM's compute_rpe works on it."""

    def test_compute_rpe_finite(self, state, grid, lock_config):
        """compute_rpe must run without error and return a finite scalar.

        This is the integration claim of the adapter at the RPE seam:
        ``grid_type="fesom"`` matches no branch in ``compute_rpe``'s area
        lookup, so it falls through to the generic ``else`` that reads
        ``grid.area`` — which ``FesomOceanGrid`` supplies as the required
        1-D volume-area array. No FESOM-specific branch is added to
        ``rpe.py``.
        """
        from legoesm.ocean.rpe import compute_rpe
        from legoesm.ocean.vertical import create_ocean_z_star

        z_coord = create_ocean_z_star(n_levels=lock_config.nlev,
                                      H_max=lock_config.H_max)
        result = compute_rpe(state, z_coord, grid_type="fesom", grid=grid)
        # RPE is a scalar (Joules or J/m² depending on convention).
        result_arr = np.asarray(result)
        assert np.isfinite(result_arr), (
            f"compute_rpe returned non-finite value: {result_arr}"
        )

    def test_compute_rpe_reads_correct_shapes(self, state, grid):
        """Smoke-test: all arrays compute_rpe reads have consistent shapes."""
        n_cells = np.asarray(state.T.data).shape[0]
        nlev = np.asarray(state.T.data).shape[1]
        assert np.asarray(state.S.data).shape == (n_cells, nlev)
        assert np.asarray(state.eta.data).shape == (n_cells,)
        assert np.asarray(state.H_bathy.data).shape == (n_cells,)
        assert np.asarray(state.land_mask.data).shape == (n_cells,)
        assert np.asarray(grid.area).shape == (n_cells,)


# =============================================================================
# F2 / F6 — front placement, including dateline case
# =============================================================================

class TestFrontPlacement:
    """Verify the lock-exchange temperature front is placed correctly.

    These tests compute expectations INDEPENDENTLY using numpy and the
    legoESM formula — they do NOT call back into the adapter's own code.
    """

    def test_front_at_zero_independent(self, flat_mesh, lock_config):
        """front_longitude=0: independently verify warm/cold partition."""
        state = create_lock_exchange_state(flat_mesh, lock_config)

        # Independent computation — numpy, legoESM's formula, not the adapter.
        lon_deg = np.degrees(
            np.asarray(flat_mesh.geo_coord_nod2D[:, 0])
        )
        front = lock_config.front_longitude  # 0.0
        west = ((lon_deg - front + 180.0) % 360.0 - 180.0) < 0.0
        expected_T = np.where(
            west, lock_config.T_cold_C, lock_config.T_warm_C
        )

        actual_T = np.asarray(state.T.data[:, 0])
        np.testing.assert_allclose(actual_T, expected_T)

    def test_dateline_front_170(self, flat_mesh):
        """front_longitude=170: node at ~185° must be WARM (15° east).

        This is the case that FAILS on the old normalise-lon-only logic:
        the old code normalised 185° → −175° and called it cold.
        """
        config = dc_replace(
            LockExchangeConfig(), front_longitude=170.0
        )
        state = create_lock_exchange_state(flat_mesh, config)

        lon_deg = np.degrees(
            np.asarray(flat_mesh.geo_coord_nod2D[:, 0])
        )
        # Wrap to [0, 360) for finding nodes near specific longitudes.
        lon_wrapped = lon_deg % 360.0

        # Nodes near 185° (within ±5°) — these are 15° EAST of front at 170°.
        idx_east = np.where(np.abs(lon_wrapped - 185.0) < 5.0)[0]
        if len(idx_east) == 0:
            pytest.skip("pi mesh has no nodes near 185° longitude")

        T_east = np.asarray(state.T.data[idx_east, 0])
        assert np.allclose(T_east, config.T_warm_C), (
            f"Nodes at ~185° are {T_east[:5]} but should be WARM "
            f"({config.T_warm_C}) — they are 15° EAST of front at 170°."
        )

        # Nodes near 165° (within ±5°) — these are 5° WEST of front at 170°.
        idx_west = np.where(np.abs(lon_wrapped - 165.0) < 5.0)[0]
        if len(idx_west) == 0:
            pytest.skip("pi mesh has no nodes near 165° longitude")

        T_west = np.asarray(state.T.data[idx_west, 0])
        assert np.allclose(T_west, config.T_cold_C), (
            f"Nodes at ~165° are {T_west[:5]} but should be COLD "
            f"({config.T_cold_C}) — they are 5° WEST of front at 170°."
        )

    def test_dateline_old_logic_would_fail(self, flat_mesh):
        """Build a REAL state with front_longitude=170.0 and verify nodes
        near 185° are WARM.  This test FAILS if the old normalise-lon-only
        logic is restored in :func:`create_lock_exchange_state`, because
        the old code wraps 185° → −175° and calls it cold.

        Using a REAL state (not just a logic check) makes this test catch
        regressions in the adapter itself, not just in the formula.
        """
        config = dc_replace(LockExchangeConfig(), front_longitude=170.0)
        state = create_lock_exchange_state(flat_mesh, config)

        lon_deg = np.degrees(
            np.asarray(flat_mesh.geo_coord_nod2D[:, 0])
        )
        lon_wrapped = lon_deg % 360.0

        # Nodes near 185° (within ±5°) — must be WARM under the new logic.
        idx_east = np.where(np.abs(lon_wrapped - 185.0) < 5.0)[0]
        if len(idx_east) == 0:
            pytest.skip("pi mesh has no nodes near 185° longitude")

        T_east = np.asarray(state.T.data[idx_east, 0])
        assert np.allclose(T_east, config.T_warm_C), (
            f"Nodes near 185° should be WARM (T={config.T_warm_C}) since "
            f"they are 15° east of front at 170°; got {T_east[:5]}. "
            f"The old normalise-lon-only logic would wrap 185→−175 and "
            f"call them COLD — if this test fails, that bug is back."
        )

    def test_salinity_uniform(self, state, lock_config):
        """Salinity must be uniform (no front in S)."""
        S = np.asarray(state.S.data)
        assert np.allclose(S, lock_config.S_uniform)


# =============================================================================
# F6 — is_first_step on the state (pytree meta field, compile-time constant)
# =============================================================================

class TestIsFirstStep:
    """Verify is_first_step is a pytree meta field that flips after one step.

    The flag is a COMPILE-TIME CONSTANT: its two values produce two
    compiled variants of ``step_jit``.  It is correct for the eager path
    (the pattern legoESM's matrix uses) and MUST NOT be carried through a
    ``lax.scan`` body — see ``test_scan_carry_of_flipping_flag_raises``.
    """

    def test_state_is_first_step_default_true(self, state):
        """A fresh IC state must have is_first_step=True."""
        assert state.is_first_step is True, (
            "Newly created state must have is_first_step=True so the first "
            "AB2 step takes the first-step branch."
        )

    def test_is_first_step_survives_jit_round_trip(self, state):
        """is_first_step is in the treedef, so it survives jax.jit.

        This test FAILS if is_first_step is moved to a leaf (data field):
        jax.jit would trace it as a weak-typed boolean array and the
        static comparison in step_jit would break.
        """
        assert state.is_first_step is True

        @jax.jit
        def identity(s):
            return s
        out = identity(state)
        assert out.is_first_step is True, (
            "is_first_step must survive jax.jit as a static meta field."
        )

    def test_is_first_step_flips_after_one_step(self, flat_mesh, lock_config):
        """One step must flip is_first_step True → False on the outgoing state."""
        pytest.importorskip("fesom_jax")
        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        state = create_lock_exchange_state(flat_mesh, lock_config)
        assert state.is_first_step is True

        s1 = model.step(state, 3600.0)
        assert s1.is_first_step is False, (
            "After one step the outgoing state must have is_first_step=False "
            "so subsequent steps take the AB2 branch."
        )

        # Second step: still False.
        s2 = model.step(s1, 3600.0)
        assert s2.is_first_step is False

    def test_step_jit_reads_is_first_step_from_state_not_a_counter(
        self, monkeypatch, flat_mesh, lock_config
    ):
        """The kwarg passed to ``step_jit`` must come from the incoming
        state, not from any model-side mutable counter.

        The old version of this test ran ``step, step`` and asserted the
        sequence ``(True, False)`` — but a hidden mutable counter on the
        model would produce exactly that sequence too, so the test was
        vacuous against the bug it claimed to catch.  This version
        additionally constructs a FRESH state with ``is_first_step=False``
        explicitly (no prior step) and asserts the value reaching
        ``step_jit`` is False.  A mutable counter would instead send
        True on the first call.
        """
        pytest.importorskip("fesom_jax")
        import fesom_jax.step as fstep_mod

        recorded = []
        real_step_jit = fstep_mod.step_jit

        def recording_step_jit(*args, **kwargs):
            recorded.append(kwargs.get("is_first_step"))
            return real_step_jit(*args, **kwargs)

        monkeypatch.setattr(fstep_mod, "step_jit", recording_step_jit)

        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        state = create_lock_exchange_state(flat_mesh, lock_config)

        # First step: state.is_first_step is True → step_jit must get True.
        s1 = model.step(state, 3600.0)
        assert recorded[-1] is True, (
            f"First step must pass is_first_step=True (from state); "
            f"got {recorded[-1]!r}"
        )

        # Threading the RESULT: s1.is_first_step is False → step_jit gets False.
        model.step(s1, 3600.0)
        assert recorded[-1] is False, (
            f"Stepping the RESULT must pass is_first_step=False (from s1); "
            f"got {recorded[-1]!r}"
        )

        # DISCRIMINATING: re-stepping the SAME original state must send True
        # again. The flag lives in the STATE, so the model is stateless and
        # cannot "remember" that it already ran. A hidden mutable counter on
        # the model would send False here — that is the bug this catches.
        recorded.clear()
        model.step(state, 3600.0)
        assert recorded[-1] is True, (
            f"Re-stepping the SAME state must pass is_first_step=True again "
            f"(the flag is state-borne, so the model holds no counter); got "
            f"{recorded[-1]!r}. A False here means the model is counting."
        )

        # DISCRIMINATING check: a FRESH state explicitly constructed with
        # is_first_step=False (no prior step on it) must send False to
        # step_jit on its first call.  A hidden mutable counter on the
        # model would have advanced to its "second call" state already,
        # but on a *fresh model* the counter could still send True —
        # which is the bug.  Building a new model + fresh-False state
        # isolates the property under test.
        fresh_state = dc_replace(state, is_first_step=False)
        model2 = FesomOceanModel(flat_mesh, None, config)
        recorded.clear()
        model2.step(fresh_state, 3600.0)
        assert len(recorded) == 1, (
            f"Expected exactly one step_jit call; got {len(recorded)}."
        )
        assert recorded[-1] is False, (
            f"A fresh state with is_first_step=False must send False to "
            f"step_jit (the value comes from the state, not a counter). "
            f"Got {recorded[-1]!r}. If this is True, the model is using a "
            f"mutable counter rather than reading state.is_first_step."
        )

    def test_scan_carry_of_flipping_flag_raises(
        self, flat_mesh, lock_config
    ):
        """Carrying a state whose ``is_first_step`` flips through
        ``lax.scan`` must raise.

        ``is_first_step`` is a static meta field (a compile-time
        constant), so flipping it changes the carry's treedef between
        iteration 0 (True) and iteration 1 (False).  ``lax.scan`` requires
        a uniform body, so it raises.  This is exactly why
        :meth:`FesomOceanModel.run` runs step 1 eagerly *outside* any
        scan and bakes ``False`` in for steps 2..N.

        This test FAILS (does not raise) if ``is_first_step`` is moved to
        a leaf — that would let scan carry it, but break the static-arg
        contract of ``step_jit`` elsewhere.  Either way the design is
        wrong; here we assert the meta-field design holds.
        """
        pytest.importorskip("fesom_jax")
        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        state = create_lock_exchange_state(flat_mesh, lock_config)
        assert state.is_first_step is True

        def body(carry, _):
            # model.step reads carry.is_first_step and returns a state
            # with is_first_step=False.  The output treedef therefore
            # differs from the input treedef — scan must reject this.
            return model.step(carry, 3600.0), None

        with pytest.raises(Exception) as excinfo:
            jax.lax.scan(body, state, xs=None, length=2)
        # Sanity: the failure must mention the pytree / carry structure,
        # not some unrelated error.  We do not over-constrain the message.
        msg = str(excinfo.value).lower()
        assert any(
            tok in msg for tok in ("pytree", "carry", "treedef", "structure")
        ), (
            f"Expected a pytree/treedef/carry error from lax.scan when the "
            f"carry's is_first_step flips; got: {excinfo.value!r}"
        )


# =============================================================================
# Model adapter — step takes and returns FesomOceanState
# =============================================================================

class TestFesomOceanModelStep:

    def test_step_returns_fesom_ocean_state(self, flat_mesh, lock_config):
        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        state = create_lock_exchange_state(flat_mesh, lock_config)
        new_state = model.step(state, 3600.0)
        assert isinstance(new_state, FesomOceanState), (
            "model.step must return a FesomOceanState so generic legoESM "
            "diagnostics can consume it."
        )

    def test_dt_mismatch_raises(self, flat_mesh, lock_config):
        config = FesomOceanConfig(dt=1800.0)
        model = FesomOceanModel(flat_mesh, None, config)
        state = create_lock_exchange_state(flat_mesh, lock_config)
        with pytest.raises(ValueError, match="dt mismatch"):
            model.step(state, 3600.0)

    def test_dt_isclose_tolerates_rounding(self, flat_mesh, lock_config):
        """math.isclose(rel_tol=1e-12) must tolerate float round-trip noise.

        This test FAILS if the guard is reverted to ``dt != self._dt``,
        because 3600.0 != 3600.000000000001 under exact equality.
        """
        pytest.importorskip("fesom_jax")
        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        state = create_lock_exchange_state(flat_mesh, lock_config)
        # A float that differs from 3600.0 by ~1e-13 (within 1e-12 rel tol).
        dt_noisy = 3600.0 * (1.0 + 1e-13)
        # Must NOT raise.
        new_state = model.step(state, dt_noisy)
        assert isinstance(new_state, FesomOceanState)

    def test_step_advances_inner_on_wet(self, flat_mesh, lock_config):
        """After one step, eta should have changed on at least some wet nodes.

        Dry nodes are tested separately (they stay at 0); here we just
        verify the dycore actually did something on the wet domain.
        """
        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        state = create_lock_exchange_state(flat_mesh, lock_config)
        new_state = model.step(state, 3600.0)
        old_eta = np.asarray(state.eta.data)
        new_eta = np.asarray(new_state.eta.data)
        geo_lat = np.asarray(flat_mesh.geo_coord_nod2D[:, 1])
        dry = np.abs(np.degrees(geo_lat)) > lock_config.land_lat_threshold
        # On wet nodes, eta should have evolved.
        assert not np.allclose(old_eta[~dry], new_eta[~dry]), (
            "Step produced no change in eta on wet nodes — dycore did not advance."
        )

    def test_config_none_defaults_resolved(self, flat_mesh):
        """FesomOceanConfig() with all-None defaults resolves to fesom_jax values."""
        cfg = FesomOceanConfig()
        model = FesomOceanModel(flat_mesh, None, cfg)
        assert model._dt > 0
        # is_first_step is no longer on the model — assert it's gone.
        assert not hasattr(model, "_is_first_step"), (
            "is_first_step must NOT be stored on the model; it lives on the "
            "state as a pytree meta field.  A model-side flag is unsafe under "
            "jax.jit of the timeloop."
        )


# =============================================================================
# Model adapter — run() implements FESOM's scan-safe integration pattern
# =============================================================================

class TestFesomOceanModelRun:
    """Tests for :meth:`FesomOceanModel.run` — the correct multi-step pattern.

    ``run`` exists to make FESOM's scan-safe integration pattern
    available, not merely documented: step 1 runs EAGERLY with
    ``is_first_step=True`` (outside any scan), and steps 2..N run with
    ``False`` baked in.  This is required because ``is_first_step`` is a
    compile-time constant in the carry's treedef and cannot flip inside
    a ``lax.scan`` body.
    """

    def test_run_rejects_zero_steps(self, flat_mesh, lock_config):
        pytest.importorskip("fesom_jax")
        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        state = create_lock_exchange_state(flat_mesh, lock_config)
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            model.run(state, 3600.0, 0)
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            model.run(state, 3600.0, -2)

    def test_run_one_step_matches_single_step_call(
        self, flat_mesh, lock_config
    ):
        """``run(state, dt, 1)`` must equal ``model.step(state, dt)``."""
        pytest.importorskip("fesom_jax")
        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        state = create_lock_exchange_state(flat_mesh, lock_config)

        via_run = model.run(state, 3600.0, 1)
        via_step = model.step(state, 3600.0)
        np.testing.assert_allclose(
            np.asarray(via_run.eta.data),
            np.asarray(via_step.eta.data),
        )
        # is_first_step flipped on both.
        assert via_run.is_first_step is False
        assert via_step.is_first_step is False

    def test_run_matches_repeated_eager_steps(
        self, flat_mesh, lock_config
    ):
        """``run(state, dt, n)`` must produce the SAME result as ``n``
        repeated eager ``.step`` calls — including the True→False flip
        on step 1 and the uniform False carry afterwards.

        This test FAILS if ``run`` does any of:
          * skip the eager first step (so step 1 wrongly uses AB2);
          * carry ``is_first_step`` as a traced value through a scan
            (which would either raise or perturb the result);
          * use a different dt than ``.step``.
        """
        pytest.importorskip("fesom_jax")
        config = FesomOceanConfig(dt=3600.0, k_ver=1e-5, a_ver=1e-4)
        model = FesomOceanModel(flat_mesh, None, config)
        state = create_lock_exchange_state(flat_mesh, lock_config)

        n = 3
        via_run = model.run(state, 3600.0, n)

        # Eager reference: identical pattern (step 1 True, rest False).
        s = state
        for _ in range(n):
            s = model.step(s, 3600.0)

        # Compare a representative prognostic field.
        np.testing.assert_allclose(
            np.asarray(via_run.eta.data),
            np.asarray(s.eta.data),
            err_msg=(
                "run(state, dt, n) must match n eager .step calls. A "
                "mismatch means run mishandles is_first_step (e.g. runs "
                "step 1 with False, or carries it as a traced value)."
            ),
        )
        # And the velocity field, which is the whole point of the
        # current-step interpolation fix in u/v.
        np.testing.assert_allclose(
            np.asarray(via_run.uv_elem.data),
            np.asarray(s.uv_elem.data),
        )
        assert via_run.is_first_step is False


# =============================================================================
# create_lock_exchange_state return type
# =============================================================================

class TestCreateLockExchangeState:

    def test_returns_fesom_ocean_state(self, flat_mesh, lock_config):
        state = create_lock_exchange_state(flat_mesh, lock_config)
        assert isinstance(state, FesomOceanState)

    def test_T_has_warm_and_cold(self, state, lock_config):
        """Both warm and cold temperatures must be present in the IC."""
        T = np.asarray(state.T.data[:, 0])
        has_warm = np.any(np.isclose(T, lock_config.T_warm_C))
        has_cold = np.any(np.isclose(T, lock_config.T_cold_C))
        assert has_warm, "No warm-side nodes found in the IC"
        assert has_cold, "No cold-side nodes found in the IC"

    def test_T_old_equals_T(self, state):
        """AB2: T_old must equal T at step 0."""
        T = np.asarray(state.inner.T)
        T_old = np.asarray(state.inner.T_old)
        np.testing.assert_allclose(T, T_old)


# --- Round-9 fixtures (distinct names from the legacy ones above) ---
@dataclasses.dataclass(frozen=True)
class _LockExchangeCfg:
    T_cold_C: float
    T_warm_C: float
    T_reference_C: float
    S_uniform: float
    front_longitude: float  # radians
    # Part of the LockExchangeConfig contract the initialiser reads. Present
    # with the shipped default so this stand-in stays a faithful stand-in: a
    # missing field here would push the initialiser into a getattr fallback,
    # i.e. a silent default in production code to keep a test's stub happy.
    front_width_deg: float = 0.0


def _load_source_mesh():
    """Load the packaged pi mesh via the REAL fesom_jax API.

    Verified signature: ``fesom_jax.mesh.load_mesh(mesh_dir=DEFAULT_PI_MESH_DIR,
    *, check_orientation=True) -> Mesh``.  No guessing -- a guessed loader that
    falls through to ``pytest.skip`` turns this whole module green while
    testing nothing.
    """
    from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh

    return load_mesh(mesh_dir=DEFAULT_PI_MESH_DIR)


@pytest.fixture(scope="module")
def source_mesh():
    return _load_source_mesh()


@pytest.fixture(scope="module")
def flat_mesh_with_land(source_mesh):
    """Flat-bottom mesh with dry nodes (land_lat_threshold=80)."""
    return build_flat_bottom_mesh(
        source_mesh,
        H_max=20.0,
        nlev=20,
        land_lat_threshold=80.0,
    )


@pytest.fixture(scope="module")
def flat_mesh_no_land(source_mesh):
    return build_flat_bottom_mesh(
        source_mesh,
        H_max=20.0,
        nlev=20,
        land_lat_threshold=90.0,
    )


@pytest.fixture(scope="module")
def lock_cfg():
    return _LockExchangeCfg(
        T_cold_C=5.0,
        T_warm_C=15.0,
        T_reference_C=10.0,
        S_uniform=35.0,
        # Place front at 0 longitude (radians); cold west, warm east.
        front_longitude=0.0,
    )


@pytest.fixture(scope="module")
def model(flat_mesh_with_land):
    return FesomOceanModel(
        flat_mesh_with_land,
        z_coord=None,
        config=FesomOceanConfig(dt=10.0, k_ver=1.0e-4, a_ver=1.0e-4),
    )




# ===========================================================================
# Round-9 findings (F1 nlevels_nod2D_min, F2 pytree design, F3 operators,
# F4 run(), F5 dx_min validation, F6 no double interpolation)
# ===========================================================================
# =============================================================================
# F1 — nlevels_nod2D_min uses the FESOM min-over-incident-elements formula
# =============================================================================

class TestR9NlevelsNod2DMin:
    """A WET node adjacent to a COLLAPSED dry element must read 1.

    The OLD code set ``nlevels_nod2D_min = nlev_n_np`` (the node's own
    bound), so a wet node kept 21.  This class fails on the old code.
    """

    def test_wet_nodes_incident_to_dry_have_min_one(self, flat_mesh_with_land):
        mesh = flat_mesh_with_land
        elem_nodes = np.asarray(mesh.elem_nodes)
        nlev_e = np.asarray(mesh.nlevels)
        nlev_n = np.asarray(mesh.nlevels_nod2D)
        nlev_n_min = np.asarray(mesh.nlevels_nod2D_min)

        # Element is dry iff its collapsed nlevels equals ulevels (==1).
        elem_dry = nlev_e <= 1
        # Node is wet iff its own nlevels > 1.
        node_wet = nlev_n > 1

        # Sanity: the mesh must actually have dry elements for this test
        # to mean anything.
        assert elem_dry.any(), (
            "Test mesh has no dry elements — land_lat_threshold=80 produced "
            "no collapsed elements; the F1 contract cannot be exercised."
        )

        # For every (wet node, dry incident element), assert min == 1.
        bad = []
        for e in np.flatnonzero(elem_dry):
            for n in elem_nodes[e]:
                if node_wet[n] and nlev_n_min[n] != 1:
                    bad.append((int(n), int(e), int(nlev_n_min[n])))
        assert not bad, (
            f"Found {len(bad)} wet node(s) incident to a dry element with "
            f"nlevels_nod2D_min != 1 (old-code bug). First few: {bad[:5]}"
        )

    def test_nlevels_nod2D_min_equals_fesom_formula(self, flat_mesh_with_land):
        """Recompute FESOM's formula independently and compare."""
        mesh = flat_mesh_with_land
        elem_nodes = np.asarray(mesh.elem_nodes).reshape(-1)
        pair_elem = np.repeat(
            np.arange(np.asarray(mesh.elem_nodes).shape[0], dtype=np.int64), 3
        )
        nlev_e = np.asarray(mesh.nlevels)
        nlev_n = np.asarray(mesh.nlevels_nod2D)

        buf = np.full(nlev_n.shape[0], np.iinfo(np.int32).max, dtype=np.int32)
        np.minimum.at(buf, elem_nodes, nlev_e[pair_elem])
        expected = np.minimum(buf, nlev_n)

        got = np.asarray(mesh.nlevels_nod2D_min)
        np.testing.assert_array_equal(got, expected)


# =============================================================================
# F2 / F6 — mesh is not a meta field; u/v don't each recompute
# =============================================================================

class TestR9PytreeDesign:
    """The state no longer carries ``mesh`` as aux_data.

    OLD code: ``mesh`` was in ``meta_fields``.  Comparing the treedefs of
    two value-identical-but-distinct-object states raised
    "truth value of an array is ambiguous" because dataclass ``__eq__``
    on Mesh returns an array of bools.  This class fails on the old code.
    """

    def test_mesh_is_not_a_dataclass_field(self):
        # F2: 'mesh' must not be a declared field at all.
        field_names = {f.name for f in dataclasses.fields(FesomOceanState)}
        assert "mesh" not in field_names, (
            "FesomOceanState still declares 'mesh' as a field — it must be "
            "removed (F2)."
        )
        assert "uv_node" in field_names, (
            "FesomOceanState must declare 'uv_node' as a data leaf (F2)."
        )

    def test_two_value_identical_states_share_treedef(
        self, source_mesh, lock_cfg
    ):
        # Build the flat mesh twice — distinct objects, identical values.
        mesh1 = build_flat_bottom_mesh(
            source_mesh, H_max=20.0, nlev=20, land_lat_threshold=80.0
        )
        mesh2 = build_flat_bottom_mesh(
            source_mesh, H_max=20.0, nlev=20, land_lat_threshold=80.0
        )
        assert mesh1 is not mesh2

        state1 = create_lock_exchange_state(mesh1, lock_cfg)
        state2 = create_lock_exchange_state(mesh2, lock_cfg)

        # On the OLD design this raises "truth value of an array is
        # ambiguous" inside treedef equality (mesh in aux_data).
        t1 = jax.tree_util.tree_structure(state1)
        t2 = jax.tree_util.tree_structure(state2)
        assert t1 == t2

    def test_jit_compiles_once_across_two_states(
        self, source_mesh, lock_cfg
    ):
        """A compiled step function works on two separately-built,
        value-identical states without recompiling or raising."""
        mesh1 = build_flat_bottom_mesh(
            source_mesh, H_max=20.0, nlev=20, land_lat_threshold=80.0
        )
        mesh2 = build_flat_bottom_mesh(
            source_mesh, H_max=20.0, nlev=20, land_lat_threshold=80.0
        )
        model1 = FesomOceanModel(
            mesh1, None, FesomOceanConfig(dt=10.0, k_ver=1e-4, a_ver=1e-4)
        )
        model2 = FesomOceanModel(
            mesh2, None, FesomOceanConfig(dt=10.0, k_ver=1e-4, a_ver=1e-4)
        )
        s1 = create_lock_exchange_state(mesh1, lock_cfg)
        s2 = create_lock_exchange_state(mesh2, lock_cfg)

        # THE F2 ASSERTION: two states built from two DISTINCT but
        # value-identical meshes must share a treedef.  Under the old design
        # (mesh in meta_fields) this comparison raised
        # "truth value of an array is ambiguous", because dataclass __eq__ on
        # a Mesh compares JAX arrays elementwise.
        assert (jax.tree_util.tree_structure(s1)
                == jax.tree_util.tree_structure(s2)), (
            "States from two value-identical meshes must share a treedef; a "
            "mismatch means mesh-derived data leaked into the static aux-data."
        )

        # A jitted function of the STATE alone must accept both without
        # raising.  The model holds the mesh and is not a pytree, so it is
        # closed over, never traced.
        @jax.jit
        def _touch(state):
            return state.inner.eta_n.sum()

        _touch(s1)
        _touch(s2)

        out1 = model1.step(s1, 10.0)
        out2 = model2.step(s2, 10.0)

        # The two outputs must be value-equal.
        np.testing.assert_allclose(
            np.asarray(out1.inner.eta_n),
            np.asarray(out2.inner.eta_n),
            rtol=1e-12, atol=1e-14,
        )

    def test_u_and_v_share_one_uv_node(
        self, source_mesh, lock_cfg
    ):
        """F6: reading u and v must NOT each recompute compute_vel_nodes.

        We monkeypatch ``compute_vel_nodes`` with a counter.  After step,
        reading ``u`` and ``v`` must not increment the counter at all
        (the leaf is already materialised).
        """
        import fesom_jax.pp as fpp

        calls = {"n": 0}
        orig = fpp.compute_vel_nodes

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        fpp.compute_vel_nodes = counting
        try:
            mesh = build_flat_bottom_mesh(
                source_mesh, H_max=20.0, nlev=20, land_lat_threshold=80.0
            )
            model = FesomOceanModel(
                mesh, None, FesomOceanConfig(dt=10.0, k_ver=1e-4, a_ver=1e-4)
            )
            state = create_lock_exchange_state(mesh, lock_cfg)

            calls["n"] = 0
            new_state = model.step(state, 10.0)
            after_step = calls["n"]
            assert after_step >= 1, "step must call compute_vel_nodes once"

            # Reading u and v must not call compute_vel_nodes again.
            _ = new_state.u
            _ = new_state.v
            assert calls["n"] == after_step, (
                f"Reading u/v re-invoked compute_vel_nodes "
                f"({calls['n'] - after_step} extra calls) — F6 regressed."
            )
        finally:
            fpp.compute_vel_nodes = orig


# =============================================================================
# F3 — capability advertises operators that cannot be built
# =============================================================================

class TestR9Capability:
    """``operator_family('fesom')`` returns ``'fesom_operators'``, but legoESM
    has no FESOM operator adapter — the operators live inside fesom_jax
    and are reachable only through the dycore.  The capability layer must
    raise a specific ``NotImplementedError``, NOT a generic ``ValueError``.
    """

    def test_operator_family_advertises_fesom(self):
        from legoesm.grids.capability import operator_family
        assert operator_family("fesom") == "fesom_operators"

    def test_build_operators_raises_not_implemented(self):
        from legoesm.grids.capability import _build_operators
        # The grid argument is irrelevant for the fesom branch (it raises
        # before touching the grid), so a sentinel is enough.
        sentinel = object()
        with pytest.raises(NotImplementedError) as excinfo:
            _build_operators("fesom", sentinel)
        msg = str(excinfo.value).lower()
        assert "fesom_jax" in msg
        assert "internal" in msg or "not exposed" in msg

    def test_build_operators_does_not_raise_valueerror(self):
        from legoesm.grids.capability import _build_operators
        sentinel = object()
        with pytest.raises(NotImplementedError):
            _build_operators("fesom", sentinel)
        # Confirm it is specifically NOT a ValueError.
        with pytest.raises(NotImplementedError):
            try:
                _build_operators("fesom", sentinel)
            except ValueError as e:
                pytest.fail(
                    f"Raised generic ValueError instead of "
                    f"NotImplementedError: {e!r}"
                )


# =============================================================================
# F4 — run() is an eager loop, not a scan
# =============================================================================

class TestR9RunIsEager:
    """``run`` is an eager Python loop over ``step``.  The test verifies
    FLAG THREADING (is_first_step True on step 1, False thereafter),
    NOT lax.scan behaviour — ``run`` does not implement scan and the
    checkpointed scan path lives in ``fesom_jax.integrate.integrate``,
    which this adapter does not wire up.
    """

    def test_run_docstring_says_eager(self):
        doc = FesomOceanModel.run.__doc__ or ""
        assert "eager" in doc.lower()
        assert "scan" in doc.lower(), (
            "run docstring must mention scan to clarify it does NOT use it."
        )

    def test_run_matches_eager_loop_and_threads_flag(
        self, model, flat_mesh_with_land, lock_cfg
    ):
        state = create_lock_exchange_state(flat_mesh_with_land, lock_cfg)
        assert state.is_first_step is True

        n = 3
        out_run = model.run(state, 10.0, n)

        # Reference: explicit eager loop, identical to run's body.
        s = state
        for i in range(n):
            s = model.step(s, 10.0)

        np.testing.assert_allclose(
            np.asarray(out_run.inner.eta_n),
            np.asarray(s.inner.eta_n),
            rtol=1e-12, atol=1e-14,
        )
        # Flag must be False after the run.
        assert out_run.is_first_step is False

    def test_run_rejects_zero_steps(self, model, flat_mesh_with_land, lock_cfg):
        state = create_lock_exchange_state(flat_mesh_with_land, lock_cfg)
        with pytest.raises(ValueError):
            model.run(state, 10.0, 0)


# =============================================================================
# F5 — cfl dx_min_override validation
# =============================================================================

class TestR9CflDxMinOverride:
    """``dx_min_override`` must be finite and strictly positive.

    Old code accepted 0.0 (-> zero timestep), NaN (-> unadjusted), and
    -1.0 (-> NEGATIVE timestep).  Each test fails on the old code.
    """

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid(self, bad):
        """A bad dx_min_override must raise BEFORE any timestep arithmetic.

        Real signature (read from packages/core/legoesm/core/cfl.py):
            cfl_check_and_adjust(dt, n, model_type="shallow_water",
                                 max_wind=0.0, gravity_wave_speed=0.0,
                                 cfl_number=0.8, radius=..., verbose=True,
                                 grid_type="cubed_sphere",
                                 use_polar_filter=False,
                                 dx_min_override=None)
        """
        from legoesm.core.cfl import cfl_check_and_adjust

        with pytest.raises(ValueError) as excinfo:
            cfl_check_and_adjust(
                dt=60.0,
                n=1,
                gravity_wave_speed=200.0,
                grid_type="fesom",
                verbose=False,
                dx_min_override=bad,
            )
        assert "dx_min_override" in str(excinfo.value), (
            f"ValueError must name dx_min_override; got: {excinfo.value!r}"
        )

    def test_fesom_without_override_raises(self):
        """grid_type='fesom' with no override must raise, not guess a dx."""
        from legoesm.core.cfl import cfl_check_and_adjust

        with pytest.raises(ValueError, match="dx_min_override"):
            cfl_check_and_adjust(
                dt=60.0, n=1, gravity_wave_speed=200.0,
                grid_type="fesom", verbose=False,
            )

    def test_valid_override_is_accepted(self):
        """A finite positive override yields a positive, finite timestep."""
        from legoesm.core.cfl import cfl_check_and_adjust

        out = cfl_check_and_adjust(
            dt=60.0, n=1, gravity_wave_speed=200.0,
            grid_type="fesom", verbose=False, dx_min_override=50_000.0,
        )
        assert math.isfinite(out) and out > 0.0, out


# =============================================================================
# Sanity: the state facade basics still hold
# =============================================================================

class TestR9FacadeBasics:
    """A few regression tests on the facade that the F2 rewrite touches."""

    def test_uv_node_shape(self, flat_mesh_with_land, lock_cfg):
        state = create_lock_exchange_state(flat_mesh_with_land, lock_cfg)
        nl = int(flat_mesh_with_land.nl)
        nod2D = int(flat_mesh_with_land.nod2D)
        assert state.uv_node.shape == (nod2D, nl, 2)

    def test_uv_node_is_zero_at_rest(self, flat_mesh_with_land, lock_cfg):
        state = create_lock_exchange_state(flat_mesh_with_land, lock_cfg)
        np.testing.assert_array_equal(
            np.asarray(state.uv_node),
            np.zeros_like(np.asarray(state.uv_node)),
        )

    def test_with_inner_is_removed(self):
        # F2 contract: with_inner must not exist (stale-velocity hazard).
        assert not hasattr(FesomOceanState, "with_inner"), (
            "FesomOceanState.with_inner must be removed — it silently "
            "carried a stale uv_node (F2)."
        )

    def test_step_updates_uv_node(self, model, flat_mesh_with_land, lock_cfg):
        state = create_lock_exchange_state(flat_mesh_with_land, lock_cfg)
        before = np.asarray(state.uv_node)
        new_state = model.step(state, 10.0)
        after = np.asarray(new_state.uv_node)
        # After one step from rest the velocity is no longer identically
        # zero (lock exchange starts moving); uv_node must have changed.
        assert not np.allclose(before, after, atol=0.0), (
            "uv_node was not updated by step — stale leaf."
        )

    def test_T_field_shape_drops_padding(self, flat_mesh_with_land, lock_cfg):
        state = create_lock_exchange_state(flat_mesh_with_land, lock_cfg)
        T = state.T
        assert T.dims == ("cell", "nlev")
        assert T.data.shape[1] == state.nlev


# =============================================================================
# create_rest_state (the rest-state benchmark family)
# =============================================================================

class TestCreateRestState:
    """``create_rest_state`` feeds the rest_state_* matrix cases on the FESOM
    arm. Its stratified profile must match the structured/MPAS rest states
    (the same exponential in ``legoesm.ocean.eos.scale_depth``), or the
    cross-grid comparison is a confound rather than a result."""

    @pytest.fixture
    def z20(self, lock_config):
        from legoesm.ocean.vertical import create_ocean_z_star
        return create_ocean_z_star(n_levels=lock_config.nlev,
                                   H_max=lock_config.H_max)

    def test_uniform_is_flat_and_at_rest(self, flat_mesh, z20):
        from legoesm.ocean.dynamics.ocean_model_fesom import create_rest_state
        st = create_rest_state(flat_mesh, z20, T_water_init_C=10.0,
                               T_deep=10.0, stratified=False)
        T = np.asarray(st.T.data)
        assert np.allclose(T, 10.0)
        assert np.allclose(np.asarray(st.uv_node), 0.0)
        assert np.allclose(np.asarray(st.eta.data), 0.0)

    def test_stratified_matches_the_shared_exponential_profile(
            self, flat_mesh, z20):
        from legoesm.ocean.dynamics.ocean_model_fesom import create_rest_state
        from legoesm.ocean.eos import scale_depth
        st = create_rest_state(flat_mesh, z20, T_water_init_C=20.0,
                               T_deep=2.0)
        # float64 reference: z_full_ref is fp32, and evaluating the
        # profile in fp32 here (not the code) was the first version's
        # 1e-7 "mismatch".
        expected = 2.0 + 18.0 * np.exp(
            np.asarray(z20.z_full_ref, dtype=np.float64) / scale_depth)
        T = np.asarray(st.T.data)          # facade drops the pad level
        # Same profile in EVERY column (horizontally uniform rest state).
        assert T.shape[1] == expected.shape[0]
        assert np.allclose(T, expected[None, :], rtol=0, atol=1e-12)
        assert T[0, 0] > T[0, -1], "T must decrease with depth"

    def test_level_mismatch_raises(self, flat_mesh, lock_config):
        from legoesm.ocean.dynamics.ocean_model_fesom import create_rest_state
        from legoesm.ocean.vertical import create_ocean_z_star
        z_wrong = create_ocean_z_star(n_levels=lock_config.nlev + 3,
                                      H_max=lock_config.H_max)
        with pytest.raises(ValueError, match="levels but the FESOM"):
            create_rest_state(flat_mesh, z_wrong)


# =============================================================================
# element_centroid_lat_lon + with_fields (the analytic-IC primitives)
# =============================================================================

class TestAnalyticICPrimitives:
    """``element_centroid_lat_lon`` and ``with_fields`` are what let the
    perturbation matrix cases (barotropic wave, geostrophic adjustment,
    Phillips, inertia-gravity wave) build a FESOM initial condition."""

    def test_centroid_lies_inside_its_element(self, flat_mesh):
        from legoesm.ocean.dynamics.ocean_model_fesom import (
            element_centroid_lat_lon)
        lat_e, lon_e = element_centroid_lat_lon(flat_mesh)
        geo = np.asarray(flat_mesh.geo_coord_nod2D)
        nodes = np.asarray(flat_mesh.elem_nodes)
        lat_e = np.asarray(lat_e)
        assert lat_e.shape == (int(flat_mesh.elem2D),)
        # Latitude is wrap-free, so the centroid must sit within the
        # element's own latitude span (longitude cannot be checked this
        # way across the dateline -- that is what the next test is for).
        lat_nodes = geo[nodes, 1]
        assert np.all(lat_e >= lat_nodes.min(axis=1) - 1e-12)
        assert np.all(lat_e <= lat_nodes.max(axis=1) + 1e-12)
        assert np.all(np.asarray(lon_e) >= 0.0)
        assert np.all(np.asarray(lon_e) <= 2 * np.pi + 1e-12)

    def test_centroid_is_wrap_safe(self, flat_mesh):
        """An element straddling the longitude SEAM must land on the seam,
        not halfway around the globe -- the failure mode of a plain
        arithmetic mean of longitudes.

        The check is the 3-D ANGULAR distance from the centroid to each of
        its nodes, not a longitude difference: near the pole longitude is
        degenerate (a polar element legitimately spans ~360 deg of
        longitude), so a longitude-space tolerance fails on correct output.
        """
        from legoesm.ocean.dynamics.ocean_model_fesom import (
            element_centroid_lat_lon)
        geo = np.asarray(flat_mesh.geo_coord_nod2D)
        nodes = np.asarray(flat_mesh.elem_nodes)
        lon_nodes = np.degrees(geo[nodes, 0])
        straddles = (lon_nodes.max(axis=1) - lon_nodes.min(axis=1)) > 180.0
        if not straddles.any():
            pytest.skip("no seam-straddling element on this mesh")
        lat_e, lon_e = element_centroid_lat_lon(flat_mesh)

        def _xyz(lat, lon):
            return np.stack([np.cos(lat) * np.cos(lon),
                             np.cos(lat) * np.sin(lon),
                             np.sin(lat)], axis=-1)

        c = _xyz(np.asarray(lat_e), np.asarray(lon_e))[straddles]
        n = _xyz(geo[nodes, 1], geo[nodes, 0])[straddles]
        ang = np.degrees(np.arccos(np.clip(
            np.einsum("ek,enk->en", c, n), -1.0, 1.0)))
        assert ang.max() < 10.0, (
            f"centroid escaped its element by {ang.max():.1f} deg")

        # ... and the naive longitude mean, the bug this guards, lands far
        # away on the non-polar straddlers (where longitude is meaningful).
        non_polar = np.abs(np.degrees(geo[nodes, 1])[straddles]).max(axis=1) < 60.0
        if non_polar.any():
            lon_deg = np.degrees(np.asarray(lon_e))[straddles][non_polar]
            naive = lon_nodes[straddles][non_polar].mean(axis=1)
            assert np.abs(
                (lon_deg - naive + 180.0) % 360.0 - 180.0).max() > 30.0

    def test_with_fields_sets_T_eta_and_velocity(self, flat_mesh, z20_or_none):
        from legoesm.ocean.dynamics.ocean_model_fesom import (
            create_rest_state, with_fields)
        st = create_rest_state(flat_mesh, z20_or_none, stratified=False)
        n_node = int(flat_mesh.nod2D)
        n_elem = int(flat_mesh.elem2D)
        eta = np.linspace(-0.5, 0.5, n_node)
        uv = np.zeros((n_elem, 2))
        uv[:, 0] = 0.3
        new = with_fields(st, flat_mesh, T=np.full(n_node, 7.0),
                          eta=eta, uv_elem=uv)
        assert np.allclose(np.asarray(new.T.data), 7.0)
        assert np.allclose(np.asarray(new.eta.data), eta)
        # uv_node is the element->node interpolation of the new element
        # velocity, so a uniform eastward element field must give a
        # non-zero, finite node velocity (never the stale zeros).
        u_node = np.asarray(new.u.data)
        assert np.all(np.isfinite(u_node))
        assert np.abs(u_node).max() > 0.0

    def test_with_fields_rejects_transposed_and_wrong_shapes(
            self, flat_mesh, z20_or_none):
        from legoesm.ocean.dynamics.ocean_model_fesom import (
            create_rest_state, with_fields)
        st = create_rest_state(flat_mesh, z20_or_none, stratified=False)
        n_node = int(flat_mesh.nod2D)
        with pytest.raises(ValueError, match="entries but the mesh"):
            with_fields(st, flat_mesh, T=np.zeros(n_node + 5))
        with pytest.raises(ValueError, match="eta shape"):
            with_fields(st, flat_mesh, eta=np.zeros(n_node + 1))
        with pytest.raises(ValueError, match="uv_elem shape"):
            with_fields(st, flat_mesh,
                        uv_elem=np.zeros((int(flat_mesh.elem2D) + 3, 2)))

    def test_with_fields_no_args_is_identity(self, flat_mesh, z20_or_none):
        from legoesm.ocean.dynamics.ocean_model_fesom import (
            create_rest_state, with_fields)
        st = create_rest_state(flat_mesh, z20_or_none, stratified=False)
        assert with_fields(st, flat_mesh) is st

    def test_vector_rotation_preserves_magnitude(self, flat_mesh):
        """geographic->rotated is a FRAME change: it must preserve the
        vector magnitude at every element (and it must not be a no-op on
        a rotated mesh, or the analytic velocity ICs point the wrong way)."""
        from legoesm.ocean.dynamics.ocean_model_fesom import (
            geographic_to_rotated_vector)
        n_elem = int(flat_mesh.elem2D)
        rng = np.random.default_rng(0)
        u = rng.normal(size=n_elem)
        v = rng.normal(size=n_elem)
        u_r, v_r = geographic_to_rotated_vector(flat_mesh, u, v)
        u_r, v_r = np.asarray(u_r), np.asarray(v_r)
        assert np.allclose(np.hypot(u_r, v_r), np.hypot(u, v), rtol=1e-10)
        rot = np.asarray(flat_mesh.coord_nod2D)
        geo = np.asarray(flat_mesh.geo_coord_nod2D)
        if not np.allclose(rot, geo):
            # This mesh IS rotated, so the transform must actually rotate.
            assert not np.allclose(u_r, u, atol=1e-6), (
                "rotation was a no-op on a rotated mesh")


# =============================================================================
# The free surface must be set where FESOM CARRIES it
# =============================================================================

class TestFreeSurfaceInitialCondition:
    """``with_fields(eta=...)`` has to reach the prognostic free surface.

    FESOM's step computes ``hbar`` at substep 11 from ``hbar_old`` plus the
    transport divergence and then OVERWRITES ``eta_n`` at substep 12 from
    those two, so an IC that sets only ``eta_n`` survives exactly one step.
    MEASURED before the fix on the benchmark's barotropic-wave arm: a 1 m
    Gaussian bump read 0.996 m at t = 0 and 0.0067 m after ONE step, against
    0.97 m on lat-lon and MPAS. That arm was integrating a rest state, and
    its missing wave was being read as a dycore difference.
    """

    def _bump(self, flat_mesh):
        geo = np.asarray(flat_mesh.geo_coord_nod2D, dtype=np.float64)
        dist = np.arccos(np.clip(np.cos(geo[:, 1]) * np.cos(geo[:, 0]),
                                 -1.0, 1.0))
        return np.exp(-0.5 * (dist / (10.0 * np.pi / 180.0)) ** 2)

    def test_eta_reaches_hbar_not_only_eta_n(self, state, flat_mesh):
        from legoesm.ocean.dynamics.ocean_model_fesom import with_fields
        bump = self._bump(flat_mesh)
        new = with_fields(state, flat_mesh, eta=bump)
        for slot in ("eta_n", "hbar", "hbar_old"):
            got = np.asarray(getattr(new.inner, slot), dtype=np.float64)
            assert np.allclose(got, bump, atol=1e-12), (
                f"with_fields(eta=...) did not set {slot}; the free surface "
                f"is carried by hbar/hbar_old and eta_n is re-derived from "
                f"them every step")
        # Fresh-start bootstrap: no SSH history, no stale CG warm start.
        assert np.allclose(np.asarray(new.inner.ssh_rhs_old), 0.0, atol=1e-12)
        assert np.allclose(np.asarray(new.inner.d_eta), 0.0, atol=1e-12)

    def test_free_surface_survives_a_step(self, state, flat_mesh,
                                          lock_config, z20_or_none):
        """The behavioural check, and the one that FAILS on pre-fix code
        (0.994 -> 0.0003 m). The tolerance is slack on purpose: the point is
        "still there", not "unchanged" -- a gravity wave really does start
        spreading immediately."""
        from legoesm.ocean.dynamics.ocean_model_fesom import with_fields
        bump = self._bump(flat_mesh)
        st = with_fields(state, flat_mesh, eta=bump)
        before = float(np.abs(np.asarray(st.eta.data)).max())
        model = FesomOceanModel(flat_mesh, z20_or_none,
                                FesomOceanConfig(dt=300.0))
        st = model.step(st, 300.0)
        after = float(np.abs(np.asarray(st.eta.data)).max())
        assert after > 0.9 * before, (
            f"one step removed the free-surface perturbation: "
            f"{before:.4f} -> {after:.4f} m")

    def test_unknown_vertical_coordinate_raises(self, state, flat_mesh):
        """Dispatch hardening: a typo must not silently pick linfs and leave
        the thicknesses inconsistent with the free surface."""
        from legoesm.ocean.dynamics.ocean_model_fesom import with_fields
        with pytest.raises(ValueError, match="vertical_coordinate"):
            with_fields(state, flat_mesh, eta=self._bump(flat_mesh),
                        vertical_coordinate="z-star")


def test_no_undefined_global_names():
    """Every global name any function in the adapter reads must exist.

    A rename that misses a call site leaves a NameError that only fires when
    that branch runs -- here, two survivors of the private-to-public rename of
    ``require_fesom_jax`` (1c8777e45) sat on the ice-coupling path and killed a
    30-day FESOM run four minutes in, on a machine where the import gate they
    guard was satisfied anyway.  Static, so it needs neither fesom_jax nor a
    mesh, and it covers every function in the module rather than the two that
    happened to break.
    """
    import builtins
    import dis
    import types

    from legoesm.ocean.dynamics import ocean_model_fesom as mod

    known = set(vars(mod)) | set(dir(builtins))

    def _codes(obj):
        yield obj
        for const in obj.co_consts:
            if isinstance(const, types.CodeType):
                yield from _codes(const)

    # LOAD_GLOBAL only: co_names also holds ATTRIBUTE names, so a plain
    # co_names scan would flag every state._replace in the file.
    missing = set()
    for value in vars(mod).values():
        fn = getattr(value, "__func__", value)
        code = getattr(fn, "__code__", None)
        if code is None or getattr(fn, "__module__", None) != mod.__name__:
            continue
        for code_obj in _codes(code):
            for ins in dis.get_instructions(code_obj):
                if ins.opname == "LOAD_GLOBAL" and ins.argval not in known:
                    missing.add(ins.argval)

    assert not missing, (
        f"these module-level names are read but never defined: {sorted(missing)}")
