"""Tests for the Phase G tendency probe harness.

Covers:

1. ``probe_latlon_cgrid`` runs end-to-end on a perturbed rest state
   and returns finite arrays for every field.
2. The momentum probe satisfies the same closure constraint that
   ``test_momentum_diagnostics_closure`` enforces — Σ PE components
   == ``total_u`` (and v) — so the probe correctly mirrors the
   production tendency path.
3. ``build_region_masks`` produces sensible region geometries on a
   global grid (e.g. the equator-band mask only fires for lat<5°,
   the mixed-layer mask only fires in the top few levels).
4. ``per_region_metrics`` returns the expected metrics with sensible
   limits — zero diff gives L=0, ±sign-match=1, perfect Pearson; a
   shape mismatch raises.
5. ``compare_probe_results`` skips face-staggered fields gracefully
   instead of raising, so the harness is usable even when the caller
   has not built face-aware masks.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.fidelity.tendency_probe import (
    LatLonProbeResult,
    build_region_masks,
    compare_probe_results,
    per_region_metrics,
    probe_latlon_cgrid,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid():
    return create_latlon_grid(n_lat=36, n_lon=72)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=10, H_max=4000.0)


@pytest.fixture
def state(grid, z_coord):
    """Rest state plus a small perturbation in u, v, T, eta so every
    momentum-diagnostic term has a non-trivial value to probe."""
    s = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    rng = np.random.default_rng(42)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u = 0.05 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.05 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    eta = 0.01 * rng.standard_normal((n_lat, n_lon))
    T_prof = 5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))
    T = T_prof[None, None, :] + 0.1 * rng.standard_normal((n_lat, n_lon, nlev))
    return s._replace(
        u=s.u.replace(data=jnp.asarray(u)),
        v=s.v.replace(data=jnp.asarray(v)),
        eta=s.eta.replace(data=jnp.asarray(eta)),
        T=s.T.replace(data=jnp.asarray(T)),
    )


@pytest.fixture
def config():
    return LatLonCGridOceanConfig.from_flat(
        A_h=1.0e4, A_v=1.0e-3,
        bottom_drag_r=0.0,
        implicit_vertical_mixing=False,   # explicit so the diag captures Av_vert
    )


# ---------------------------------------------------------------------------
# 0. Vertical-mixing (NEMO zdf) tracer tendency — the field the NEMO
#    tendency-match compares against ``ttrd_zdf``
# ---------------------------------------------------------------------------


def test_zdf_tendency_zero_without_vertical_mixing(state, grid, z_coord, config):
    """scheme="none" (the Veros-ACC probe default) leaves dT_zdf/dS_zdf exactly
    zero, so the existing tier-2 comparisons are unchanged by the new field."""
    result = probe_latlon_cgrid(state, grid, z_coord, config)
    assert jnp.all(result.dT_zdf == 0.0)
    assert jnp.all(result.dS_zdf == 0.0)
    assert result.dT_zdf.shape == state.T.data.shape
    assert result.dS_zdf.shape == state.S.data.shape


def test_zdf_tendency_active_with_constant_closure(state, grid, z_coord):
    """With an ACTIVE vertical-mixing closure the probe returns a finite,
    non-trivial dT_zdf — the legoESM analogue of NEMO ``ttrd_zdf`` — and it
    equals the shared implicit-diffusion increment (T_new-T_old)/dt, i.e. the
    SAME helper the production solve uses (no re-derived numerics)."""
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing import (
        build_dz_half, compute_vertical_K_profiles,
        implicit_vertical_diffusion_ocean,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig, ConstantVerticalMixingConfig,
    )
    from legoesm.ocean.vertical import compute_ocean_jacobian

    # DISTINGUISHABLE K_v vs A_v: if compute_vertical_K_profiles' (K_v, A_v)
    # return order were swapped, the probe would diffuse tracers with the
    # VISCOSITY and the reconstruction below would not match (codex LOW).
    vm = VerticalMixingConfig(
        scheme="constant",
        constant=ConstantVerticalMixingConfig(K_v=1.0e-3, A_v=7.0e-2))
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e4, A_v=1.0e-3, bottom_drag_r=0.0,
        implicit_vertical_mixing=False,
        physics=OceanPhysicsConfig(vertical_mixing=vm),
    )
    dt_tr = 300.0
    result = probe_latlon_cgrid(state, grid, z_coord, cfg, dt=dt_tr)
    assert jnp.all(jnp.isfinite(result.dT_zdf))
    # non-trivial: the initial state has vertical structure -> mixing acts
    assert float(jnp.max(jnp.abs(result.dT_zdf))) > 0.0
    # matches the shared-helper construction exactly — for BOTH T and S, and
    # with K_v (1e-3) != A_v (7e-2) a swapped return order would fail here.
    K_v, A_v = compute_vertical_K_profiles(state, z_coord, None, cfg.physics)
    assert float(jnp.max(K_v)) == pytest.approx(1.0e-3, rel=1e-6), (
        "compute_vertical_K_profiles must return (K_v, A_v) in that order")
    assert float(jnp.max(A_v)) == pytest.approx(7.0e-2, rel=1e-6)
    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    dz_cell = z_coord.dz_ref * J[:, :, jnp.newaxis]
    dz_half = build_dz_half(dz_cell)
    mask3 = state.land_mask.data[:, :, None]
    T_new = implicit_vertical_diffusion_ocean(
        state.T.data, K_v, dz_cell, dz_half, dt_tr)
    S_new = implicit_vertical_diffusion_ocean(
        state.S.data, K_v, dz_cell, dz_half, dt_tr)
    want_T = (T_new - state.T.data) / dt_tr * mask3
    want_S = (S_new - state.S.data) / dt_tr * mask3
    assert jnp.allclose(result.dT_zdf, want_T, rtol=1e-10, atol=1e-14)
    assert jnp.allclose(result.dS_zdf, want_S, rtol=1e-10, atol=1e-14)


# ---------------------------------------------------------------------------
# 1. End-to-end probe runs and returns finite arrays
# ---------------------------------------------------------------------------


def test_probe_runs_end_to_end(state, grid, z_coord, config):
    result = probe_latlon_cgrid(state, grid, z_coord, config)
    assert isinstance(result, LatLonProbeResult)
    for name in result._fields:
        arr = getattr(result, name)
        assert jnp.all(jnp.isfinite(arr)), (
            f"Probe field {name!r} contains non-finite values."
        )


def test_probe_shapes_match_state(state, grid, z_coord, config):
    """Velocity-tendency arrays sit on u/v faces; tracer-tendency arrays
    and the EOS density sit at cell centres."""
    result = probe_latlon_cgrid(state, grid, z_coord, config)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u_shape = (n_lat, n_lon + 1, nlev)
    v_shape = (n_lat + 1, n_lon, nlev)
    cell_shape = (n_lat, n_lon, nlev)
    assert result.pgf_ke_u.shape == u_shape
    assert result.pgf_ke_v.shape == v_shape
    assert result.coriolis_u.shape == u_shape
    assert result.coriolis_v.shape == v_shape
    assert result.dT_dt_total.shape == cell_shape
    assert result.dS_dt_total.shape == cell_shape
    assert result.rho.shape == cell_shape


# ---------------------------------------------------------------------------
# 2. Momentum closure: Σ PE components == total (excludes Coriolis)
# ---------------------------------------------------------------------------


def test_momentum_closure_excludes_coriolis(state, grid, z_coord, config):
    """``total_u`` is defined as the PE tendency BEFORE Coriolis is
    applied; the per-process sum (excluding Coriolis) must equal
    ``total_u`` to machine precision. This mirrors
    ``test_momentum_diagnostics_closure``."""
    result = probe_latlon_cgrid(state, grid, z_coord, config)
    # Components present in MomentumTendencyDiagnostics: everything in
    # LatLonProbeResult except coriolis_{u,v}, total_{u,v},
    # dT_dt_total, dS_dt_total, rho, and the v-mirror of each
    # momentum-u field. Build the sum directly per axis.
    sum_u = (
        result.pgf_ke_u + result.vortcor_u + result.vertadv_u
        + result.ah_lap_u + result.bh_bilap_u + result.botdrag_u
        + result.av_vert_u + result.phys_u
    )
    sum_v = (
        result.pgf_ke_v + result.vortcor_v + result.vertadv_v
        + result.ah_lap_v + result.bh_bilap_v + result.botdrag_v
        + result.av_vert_v + result.phys_v
    )
    np.testing.assert_allclose(
        np.asarray(sum_u), np.asarray(result.total_u),
        atol=1e-12, rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(sum_v), np.asarray(result.total_v),
        atol=1e-12, rtol=1e-12,
    )


def test_coriolis_separate_from_total(state, grid, z_coord, config):
    """Coriolis is NOT included in ``total_u`` (per the diagnostic
    contract). At a non-rest state, ``coriolis_u`` must be non-zero
    so the harness actually probes it."""
    result = probe_latlon_cgrid(state, grid, z_coord, config)
    assert float(jnp.max(jnp.abs(result.coriolis_u))) > 1e-10
    assert float(jnp.max(jnp.abs(result.coriolis_v))) > 1e-10


# ---------------------------------------------------------------------------
# 3. Region masks
# ---------------------------------------------------------------------------


def test_region_masks_basic_shapes(state, grid, z_coord):
    masks = build_region_masks(grid, z_coord, state)
    cell_shape = (grid.n_lat, grid.n_lon, z_coord.n_levels)
    for name in masks._fields:
        m = getattr(masks, name)
        assert m.shape == cell_shape, f"mask {name!r} shape mismatch"
        # Boolean / 0-1 valued
        m_np = np.asarray(m)
        assert m_np.dtype == bool or set(np.unique(m_np)).issubset({0, 1})


def test_equator_mask_only_low_latitudes(state, grid, z_coord):
    masks = build_region_masks(grid, z_coord, state, equator_lat_deg=5.0)
    lat_deg = np.degrees(np.asarray(grid.lat))
    eq_np = np.asarray(masks.equator)
    # Any latitude row that is in the equator mask must be |lat| <= 5
    for i, lat in enumerate(lat_deg):
        if eq_np[i].any():
            assert abs(lat) <= 5.0 + 1e-9, (
                f"Equator mask fires at lat={lat:.2f}° > 5°"
            )


def test_mixed_layer_mask_only_shallow(state, grid, z_coord):
    masks = build_region_masks(grid, z_coord, state, mixed_layer_depth_m=100.0)
    ml_np = np.asarray(masks.mixed_layer)
    dz = np.asarray(z_coord.dz_ref)
    z_centre = np.cumsum(dz) - 0.5 * dz
    for k, z in enumerate(z_centre):
        if ml_np[:, :, k].any():
            assert z <= 100.0 + 1e-9, (
                f"Mixed-layer mask fires at depth {z:.1f} m > 100 m"
            )


def test_interior_and_boundary_partition_wet(state, grid, z_coord):
    """interior ∪ boundary == wet (disjoint partition of ocean cells)."""
    masks = build_region_masks(grid, z_coord, state)
    union = np.asarray(masks.interior) | np.asarray(masks.boundary)
    np.testing.assert_array_equal(union, np.asarray(masks.wet))
    intersect = np.asarray(masks.interior) & np.asarray(masks.boundary)
    assert not intersect.any(), "interior and boundary overlap"


# ---------------------------------------------------------------------------
# 4. Per-region metrics
# ---------------------------------------------------------------------------


def test_per_region_zero_diff_gives_perfect_match(state, grid, z_coord):
    masks = build_region_masks(grid, z_coord, state)
    a = jnp.asarray(np.random.default_rng(7).standard_normal(masks.wet.shape))
    metrics = per_region_metrics(a, a, masks)
    for region in metrics:
        m = metrics[region]
        if m["n_cells"] == 0:
            continue
        assert m["L1"] == 0.0
        assert m["L2"] == 0.0
        assert m["Linf"] == 0.0
        assert m["sign_match"] == 1.0
        assert abs(m["pattern_corr"] - 1.0) < 1e-12


def test_per_region_shape_mismatch_raises(state, grid, z_coord):
    masks = build_region_masks(grid, z_coord, state)
    a = jnp.zeros(masks.wet.shape)
    b = jnp.zeros((masks.wet.shape[0] + 1,) + masks.wet.shape[1:])
    with pytest.raises(ValueError, match="shape"):
        per_region_metrics(a, b, masks)


def test_per_region_sign_match_detects_flipped_sign(state, grid, z_coord):
    masks = build_region_masks(grid, z_coord, state)
    rng = np.random.default_rng(11)
    a = jnp.asarray(rng.standard_normal(masks.wet.shape))
    flipped = -a
    metrics = per_region_metrics(a, flipped, masks)
    for region, m in metrics.items():
        if m["n_cells"] == 0:
            continue
        # All cells have opposite sign ⇒ sign_match ≈ 0
        # (allowing one tied-at-zero cell to land on either side).
        assert m["sign_match"] < 0.05, (
            f"region={region!r} sign_match={m['sign_match']} expected ~0"
        )


# ---------------------------------------------------------------------------
# 5. compare_probe_results: cell-centred fields compared, face fields skipped
# ---------------------------------------------------------------------------


def test_compare_probe_results_skips_face_fields_gracefully(state, grid, z_coord, config):
    """Cell-centre fields (rho, dT_dt_total, dS_dt_total) get compared;
    face-staggered fields (pgf_ke_u/v, coriolis_u/v, ...) are reported
    as ``_skipped`` so the harness is usable with cell-centre masks
    even before face-aware masks are wired."""
    result = probe_latlon_cgrid(state, grid, z_coord, config)
    masks = build_region_masks(grid, z_coord, state)
    cmp = compare_probe_results(result, result, masks)

    # rho is at cell centres — must be a real metrics dict
    assert "rho" in cmp
    assert "interior" in cmp["rho"]
    assert cmp["rho"]["interior"]["L2"] == 0.0

    # dT_dt_total / dS_dt_total are at cell centres — real metrics
    assert cmp["dT_dt_total"]["interior"]["L2"] == 0.0
    assert cmp["dS_dt_total"]["interior"]["L2"] == 0.0

    # pgf_ke_u sits on u-faces — must be skipped with a clear reason.
    assert "_skipped" in cmp["pgf_ke_u"]
    assert "shape" in cmp["pgf_ke_u"]["_skipped"]["reason"].lower()


# ---------------------------------------------------------------------------
# 6. GM/Redi probe == production (#1317): the probe must thread the SAME
#    kappa_redi_override production always computes, and support an optional
#    (T, S) override for callers whose oracle's iso operator lives at a
#    different time level than the state's primary T/S (NEMO leap-frog: iso
#    is evaluated on the *before* level, advection on *now* — see the
#    ``gm_redi_tracer_state`` docstring in tendency_probe.py).
# ---------------------------------------------------------------------------


@pytest.fixture
def gm_redi_config():
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    gm = GMRediConfig(
        kappa_GM=0.0, kappa_Redi=1.0e3,
        kappa_redi_lat_scaling=True,   # NEMO nn_aht_ijk_t=20 cos(lat) scaling
        slope_scheme="nemo_iso_lap",
    )
    return LatLonCGridOceanConfig.from_flat(
        A_h=0.0, A_v=1.0e-3, K_h=0.0, K_v=0.0,
        bottom_drag_r=0.0, gm_redi=gm,
    )


def test_gm_redi_probe_threads_kappa_redi_cos_lat_override(
    state, grid, z_coord, gm_redi_config,
):
    """``kappa_redi_lat_scaling=True`` (the DINO nemo_dino_kamm_mlf recipe)
    means kappa_Redi is NOT the flat scalar the probe would use if it
    silently dropped ``kappa_redi_override`` — reproduce production's own
    ``_static_kappa_redi_override`` + ``gm_redi_tracer_tendency_latlon`` call
    verbatim and assert the probe matches it bit-for-bit. This is the actual
    #1317 defect: the probe used to call ``gm_redi_tracer_tendency_latlon``
    WITHOUT ``kappa_redi_override``, silently running Redi at the (wrong,
    non-cos-scaled) equator-value kappa everywhere."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _static_kappa_redi_override,
    )
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        gm_redi_tracer_tendency_latlon,
    )

    result = probe_latlon_cgrid(
        state, grid, z_coord, gm_redi_config,
        gm_redi_tracer_state=(state.T.data, state.S.data),
    )

    kappa_redi_override, kappa_redi_v_override = _static_kappa_redi_override(
        gm_redi_config.gm_redi, grid)
    assert kappa_redi_override is not None, "fixture must exercise the cos-lat path"
    dT_expected, dS_expected = gm_redi_tracer_tendency_latlon(
        state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
        grid, z_coord, gm_redi_config.gm_redi,
        eos=gm_redi_config.eos, eos_linear=gm_redi_config.eos_linear,
        mask=state.land_mask.data,
        u_mask=state.u_mask.data, v_mask=state.v_mask.data,
        rho_0=gm_redi_config.constants.rho_0, g=gm_redi_config.constants.g,
        kappa_redi_override=kappa_redi_override,
        kappa_redi_v_override=kappa_redi_v_override,
        dt=300.0,
    )
    np.testing.assert_allclose(
        np.asarray(result.dT_gm_redi), np.asarray(dT_expected),
        atol=1e-14, rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(result.dS_gm_redi), np.asarray(dS_expected),
        atol=1e-14, rtol=1e-12,
    )

    # Sanity: the fix is not a no-op — omitting kappa_redi_override (the old,
    # buggy behaviour) gives a materially different tendency on this cos-lat
    # recipe, so a regression back to "no override" would be caught above.
    dT_no_override, _ = gm_redi_tracer_tendency_latlon(
        state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
        grid, z_coord, gm_redi_config.gm_redi,
        eos=gm_redi_config.eos, eos_linear=gm_redi_config.eos_linear,
        mask=state.land_mask.data,
        u_mask=state.u_mask.data, v_mask=state.v_mask.data,
        rho_0=gm_redi_config.constants.rho_0, g=gm_redi_config.constants.g,
        dt=300.0,
    )
    assert float(jnp.max(jnp.abs(dT_expected - dT_no_override))) > 1e-10


def test_gm_redi_tracer_state_override_changes_iso_bucket_only(
    state, grid, z_coord, gm_redi_config,
):
    """``gm_redi_tracer_state`` must retarget ONLY the GM/Redi (iso) bucket —
    perturbing it must leave every momentum/advection-adjacent probe field
    (which reads ``state.T``/``state.S`` directly, e.g. ``rho``) unchanged,
    and must change ``dT_gm_redi``/``dS_gm_redi`` (and hence the folded
    ``dT_dt_total``) relative to the default (``state.T``/``state.S``)."""
    baseline = probe_latlon_cgrid(state, grid, z_coord, gm_redi_config)

    rng = np.random.default_rng(3)
    T_before = np.asarray(state.T.data) + 0.3 * rng.standard_normal(state.T.data.shape)
    S_before = np.asarray(state.S.data)
    perturbed = probe_latlon_cgrid(
        state, grid, z_coord, gm_redi_config,
        gm_redi_tracer_state=(jnp.asarray(T_before), jnp.asarray(S_before)),
    )

    # rho is computed from state.T/state.S directly (not gm_redi_tracer_state)
    # -> must be untouched by the override.
    np.testing.assert_array_equal(np.asarray(baseline.rho), np.asarray(perturbed.rho))

    # The iso bucket must actually respond to the override.
    assert float(jnp.max(jnp.abs(
        perturbed.dT_gm_redi - baseline.dT_gm_redi))) > 1e-10
    # ... and it must be folded into dT_dt_total (dT_dt_total = dT_dt + dT_gm).
    np.testing.assert_allclose(
        np.asarray(perturbed.dT_dt_total - perturbed.dT_gm_redi),
        np.asarray(baseline.dT_dt_total - baseline.dT_gm_redi),
        atol=1e-14, rtol=1e-12,
    )


def test_gm_redi_tracer_state_default_none_is_backward_compatible(
    state, grid, z_coord, gm_redi_config,
):
    """``gm_redi_tracer_state=None`` (the default, used by every pre-existing
    caller: compare_tendencies_acc.py, budget_{pointwise,fullframe}.py,
    momentum_budget_diff.py, poison_gate.py) must be byte-identical to
    explicitly passing ``(state.T.data, state.S.data)``."""
    default = probe_latlon_cgrid(state, grid, z_coord, gm_redi_config)
    explicit = probe_latlon_cgrid(
        state, grid, z_coord, gm_redi_config,
        gm_redi_tracer_state=(state.T.data, state.S.data),
    )
    np.testing.assert_array_equal(
        np.asarray(default.dT_gm_redi), np.asarray(explicit.dT_gm_redi))
    np.testing.assert_array_equal(
        np.asarray(default.dS_gm_redi), np.asarray(explicit.dS_gm_redi))
