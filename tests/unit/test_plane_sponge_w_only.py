"""SAM-faithful w-only top sponge (D4, iter-47).

SAM ``damping.f90`` damps the VERTICAL velocity ONLY (``w/(1+taudamp)``);
u/v/θ/ρ are untouched, so the sponge absorbs gravity waves while leaving the
anvil-level horizontal wind + thermodynamics intact. ``sponge_w_only=True``
makes the plane dycore match that; the default (False) damps all 5 fields.

These tests isolate the sponge term: with the SAME state, the only difference
between the two configs is the u/v/θ/ρ sponge damping, so the tendency
difference equals exactly ``sponge_profile · field``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    sponge_profile, make_flat_plane_terrain_metric, make_rest_state,
    plane_compressible_euler_slow_tendencies,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def _setup(sponge_w_only):
    grid = create_plane_grid(nx=8, ny=8, nlev=12, dx=1_000.0, dy=1_000.0,
                             dtype=jnp.float64)
    hc = create_height_coordinate(grid.nlev, H=12_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=4_800.0,   # base at 0.6·H
        sponge_w_only=sponge_w_only,
        semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
    )
    return grid, hc, tm, cfg


def _perturbed_state(grid, hc):
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # sheared mean wind + θ'/ρ' perturbations so every sponged field is nonzero
    u = 2.0 + 0.001 * jnp.broadcast_to(
        hc.z_full[None, None, :], (grid.ny, grid.nx, grid.nlev))
    return state._replace(
        u=state.u.replace(data=u),
        v=state.v.replace(data=jnp.full_like(u, -1.5)),
        theta_prime=state.theta_prime.replace(data=jnp.full_like(u, 0.3)),
        rho_prime=state.rho_prime.replace(data=jnp.full_like(u, 0.002)),
        w=state.w.replace(data=jnp.full(
            (grid.ny, grid.nx, grid.nlev + 1), 0.1)),
    )


def test_w_only_removes_uvtr_sponge_keeps_w():
    """tend(w_only) − tend(5-field) = +sponge·field for u/v/θ/ρ; dw_dt
    identical (w damped in BOTH)."""
    grid, hc, tm, cfg_5 = _setup(sponge_w_only=False)
    _, _, _, cfg_w = _setup(sponge_w_only=True)
    state = _perturbed_state(grid, hc)
    t5 = plane_compressible_euler_slow_tendencies(state, grid, hc, tm, cfg_5)
    tw = plane_compressible_euler_slow_tendencies(state, grid, hc, tm, cfg_w)
    sponge_full = sponge_profile(
        hc.z_full, hc.H, cfg_5.sponge_width, cfg_5.sponge_coeff)
    # the removed sponge term = sponge·field (positive where the profile bites)
    np.testing.assert_allclose(
        np.asarray(tw.du_dt.data - t5.du_dt.data),
        np.asarray(sponge_full * state.u.data), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(tw.dtheta_prime_dt.data - t5.dtheta_prime_dt.data),
        np.asarray(sponge_full * state.theta_prime.data), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(tw.drho_prime_dt.data - t5.drho_prime_dt.data),
        np.asarray(sponge_full * state.rho_prime.data), rtol=0.0, atol=1e-12)
    # w damping unchanged
    np.testing.assert_array_equal(
        np.asarray(tw.dw_dt.data), np.asarray(t5.dw_dt.data))


def test_w_only_preserves_upper_level_wind_tendency():
    """The physical point: in the sponge layer the w-only config applies NO
    Rayleigh drag to the horizontal wind, so the anvil-level wind is not
    spun down toward zero (unlike the 5-field sponge)."""
    grid, hc, tm, cfg_5 = _setup(sponge_w_only=False)
    _, _, _, cfg_w = _setup(sponge_w_only=True)
    state = _perturbed_state(grid, hc)
    t5 = plane_compressible_euler_slow_tendencies(state, grid, hc, tm, cfg_5)
    tw = plane_compressible_euler_slow_tendencies(state, grid, hc, tm, cfg_w)
    sponge_full = np.asarray(sponge_profile(
        hc.z_full, hc.H, cfg_5.sponge_width, cfg_5.sponge_coeff))
    ktop = int(np.argmax(sponge_full > 0))   # first sponged level from the top
    # 5-field config drags u toward zero there (negative tendency contribution);
    # w-only does not → the two differ at the top, agree below the sponge.
    assert float(sponge_full[ktop]) > 0.0
    du5_top = float(np.asarray(t5.du_dt.data)[0, 0, ktop])
    duw_top = float(np.asarray(tw.du_dt.data)[0, 0, ktop])
    assert duw_top > du5_top   # w-only lacks the −sponge·u drag (less negative)
    # below the sponge (lowest level) both identical
    np.testing.assert_allclose(
        np.asarray(t5.du_dt.data)[..., -1],
        np.asarray(tw.du_dt.data)[..., -1], rtol=0.0, atol=1e-13)


def test_w_only_config_default_false():
    """Back-compat: default keeps the 5-field sponge + sin² profile."""
    assert CompressibleEulerConfig().sponge_w_only is False
    assert CompressibleEulerConfig().sponge_profile_shape == "sin2"


def test_sam_rational_profile_ramps_faster_than_sin2():
    """codex iter-47 E: SAM ``zzz/(1+zzz)`` (zzz=100·frac²) ramps FAST after the
    base (≈half strength by frac=0.1), vs sin²'s gentle, upward-shifted taper.
    Endpoints match (0 at base, ~coeff at top); rational ≥ sin² in the lower
    sponge."""
    H, width, coeff = 20_000.0, 8_000.0, 0.05
    base = H - width
    z = jnp.asarray([base, base + 0.1 * width, base + 0.5 * width, H])
    sin2 = np.asarray(sponge_profile(z, H, width, coeff, shape="sin2"))
    rat = np.asarray(sponge_profile(z, H, width, coeff, shape="sam_rational"))
    # base = 0 for both; top ≈ coeff for both
    assert sin2[0] == 0.0 and rat[0] == 0.0
    np.testing.assert_allclose(rat[3], coeff * 100.0 / 101.0, rtol=1e-12)
    # at frac=0.1 the rational is ~half strength; sin² is ~2.4% — much gentler
    np.testing.assert_allclose(rat[1], coeff * 0.5, rtol=1e-12)
    assert rat[1] > 10.0 * sin2[1]
    # rational dominates the lower sponge (faster ramp)
    assert rat[1] > sin2[1] and rat[2] > sin2[2]


def test_unknown_sponge_shape_raises():
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import sponge_profile
    import pytest
    with pytest.raises(ValueError, match="Unknown sponge profile shape"):
        sponge_profile(jnp.zeros(3), 1000.0, 500.0, 0.05, shape="bogus")


def test_crm_scripts_set_sam_faithful_sponge():
    """codex iter-47 G: guard that the CRM drivers reach the kernel with the
    SAM-faithful sponge (w-only + rational taper). A regression dropping either
    would silently un-faithful the top damping."""
    from pathlib import Path
    # namespace-safe repo root (legoesm is a PEP-420 namespace pkg, no __file__)
    repo = Path(__file__).resolve().parents[2]
    for script in ("run_gate_plane.py", "run_lba_plane.py",
                   "run_rcemip_plane.py"):
        src = (repo / "scripts" / "run" / script).read_text()
        assert "sponge_w_only=True" in src, f"{script}: w-only sponge"
        assert 'sponge_profile_shape="sam_rational"' in src, (
            f"{script}: SAM rational sponge taper")
