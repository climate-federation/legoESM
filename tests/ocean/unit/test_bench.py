"""Unit tests for the BENCH experiment (Irrmann et al. 2022 port).

Covers, with planted-violation controls where a GATE is asserted (the
repo rule: a check that cannot fail does not check):

* ``build_z2d_ramp``: range, per-point uniqueness, hemisphere mirroring;
* ICs on the latlon grid: finiteness, bounds, per-point T uniqueness,
  uniform z* grid (NEMO usrdef_zgr), flat bottom / all-ocean mask;
* ICs on the synthetic tripole grid: finiteness, uniqueness;
* the stability validator: NaN / overspeed / overshot-eta planted
  violations each FAIL the gate (the control that proves the gate fires);
* the model-config factory: recipe identity + forcing scheme none.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from legoesm.ocean.experiments import bench  # noqa: E402

# ---------------------------------------------------------------------------
# z2d ramp (usrdef_istate.F90:44-54)
# ---------------------------------------------------------------------------


class TestZ2dRamp:
    def test_range(self):
        z2d = np.asarray(bench.build_z2d_ramp(16, 32))
        assert z2d.shape == (16, 32)
        assert np.all(z2d >= -0.11)
        assert np.all(z2d <= 0.11)

    def test_unique_per_point(self):
        # The MPI/sharding-bug detector: every point carries a unique value.
        z2d = np.asarray(bench.build_z2d_ramp(24, 48))
        assert np.unique(z2d.round(12)).size == z2d.size

    def test_hemisphere_mirroring(self):
        # NEMO: z2d(j=Nj0/2 row) ≈ -z2d(j=Nj0/2 + 1 row) — the two halves
        # sweep the ramp in opposite directions across the equator split.
        z2d = np.asarray(bench.build_z2d_ramp(16, 32))
        south_row = z2d[7]
        north_row = z2d[8]
        # The mirrored sweep: sum of the two rows is constant per column
        # (0.1*(p-0.5) + 0.1*(1.5-p') with the NEMO index construction is
        # constant per column up to the row-pairing offset).
        sums = south_row + north_row
        assert np.allclose(sums, sums[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Vertical grid (usrdef_zgr.F90:139-165)
# ---------------------------------------------------------------------------


class TestUniformZStar:
    def test_uniform_levels_and_depth(self):
        z = bench.bench_uniform_z_star(10, 4000.0)
        dz = np.asarray(z.dz_ref)
        assert np.allclose(dz, 400.0)
        assert float(z.H_max) == 4000.0
        # Interfaces: 0, -400, ..., -4000 (NEMO: depw(k) = (k-1)*zd)
        zh = np.asarray(z.z_half_ref)
        assert np.allclose(zh, -400.0 * np.arange(11.0))


# ---------------------------------------------------------------------------
# ICs on the latlon grid
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def latlon_setup():
    from legoesm.grids.latlon import create_latlon_grid

    grid = create_latlon_grid(12, 24)
    z = bench.bench_uniform_z_star(6, 4000.0)
    cfg = bench.BenchConfig(n_levels=6, H_max=4000.0)
    state = bench.create_initial_conditions("latlon", grid, z, cfg)
    return grid, z, cfg, state


class TestLatlonICs:
    def test_finite(self, latlon_setup):
        _, _, _, st = latlon_setup
        for name in ("T", "S", "u", "v", "eta"):
            assert np.all(np.isfinite(np.asarray(getattr(st, name).data))), name

    def test_all_ocean_flat_bottom(self, latlon_setup):
        _, _, _, st = latlon_setup
        mask = np.asarray(st.land_mask.data)
        assert np.all(mask == 1.0)  # NEMO BENCH: no land
        H = np.asarray(st.H_bathy.data)
        assert np.allclose(H, 4000.0)  # usrdef_zgr: flat bottom

    def test_T_unique_per_level_and_per_column(self, latlon_setup):
        # The MPI/sharding-bug detection property: every horizontal
        # point carries a unique value AT EACH LEVEL (halo exchanges are
        # 2-D per level), and every column carries distinct values
        # across levels (the stratification trend).  Global 3-D
        # uniqueness is NOT asserted: the level ranges overlap by
        # construction (NEMO's T = 20*z2d - 1 - 0.5*f has the same
        # structure — the paper's uniqueness claim is the per-point /
        # per-level one, usrdef_istate.F90:44-54's doubled numerator).
        _, _, _, st = latlon_setup
        T = np.asarray(st.T.data)
        for k in range(T.shape[2]):
            level = T[:, :, k].round(12)
            assert np.unique(level).size == level.size, f"level {k}"
        j0, i0 = 3, 5
        column = T[j0, i0, :]
        assert np.unique(column.round(12)).size == column.size

    def test_eta_unique_and_bounded(self, latlon_setup):
        _, z, cfg, st = latlon_setup
        eta = np.asarray(st.eta.data)
        # usrdef_istate_ssh: +/- ssh_amplitude/2 about zero (0.1*(0.5-p)).
        assert np.all(np.abs(eta) <= cfg.ssh_amplitude * 0.5 + 1e-12)
        assert np.unique(eta.round(12)).size == eta.size

    def test_velocity_bounded_by_ic_amplitudes(self, latlon_setup):
        _, _, cfg, st = latlon_setup
        u = np.asarray(st.u.data)
        v = np.asarray(st.v.data)
        assert np.all(np.abs(u) <= cfg.u_z2d_coeff * 0.1 + 1e-12)
        assert np.all(np.abs(v) <= cfg.v_z2d_coeff * 0.1 + 1e-12)

    def test_light_stratification_stable(self, latlon_setup):
        # T decreases, S increases with depth (the light stratification
        # that "keeps the vertical stability of the model", paper 2.2.1)
        # at the mean-over-points level.
        _, _, _, st = latlon_setup
        T = np.asarray(st.T.data).mean(axis=(0, 1))
        S = np.asarray(st.S.data).mean(axis=(0, 1))
        assert T[0] > T[-1]
        assert S[0] < S[-1]


# ---------------------------------------------------------------------------
# ICs on the synthetic tripole grid
# ---------------------------------------------------------------------------


class TestTripoleICs:
    def test_finite_and_unique(self):
        from legoesm.grids.tripole import create_synthetic_tripole

        grid = create_synthetic_tripole(n_lat=16, n_lon=32)
        assert grid.fold.is_active
        z = bench.bench_uniform_z_star(5, 4000.0)
        cfg = bench.BenchConfig(n_levels=5, H_max=4000.0)
        st = bench.create_initial_conditions("tripole", grid, z, cfg)
        for name in ("T", "S", "u", "v", "eta"):
            arr = np.asarray(getattr(st, name).data)
            assert np.all(np.isfinite(arr)), name
        # Per-level 2-D uniqueness (the bug-detection property).
        T = np.asarray(st.T.data)
        for k in range(T.shape[2]):
            level = T[:, :, k].round(12)
            assert np.unique(level).size == level.size, f"level {k}"


# ---------------------------------------------------------------------------
# Forcing / model config
# ---------------------------------------------------------------------------


class TestForcingsAndConfig:
    def test_forcing_is_none_scheme(self):
        physics = bench.create_forcings("latlon", None, bench.BenchConfig())
        assert physics.surface_forcing.scheme == "none"  # usrdef_sbc zeros

    def test_model_config_recipe_identity(self):
        from legoesm.ocean.recipes import get_recipe

        cfg = bench.BenchConfig()
        mc = bench.bench_model_config(cfg)
        want = get_recipe("legoesm_nemo_like_v1", "latlon")
        for k, v in want.items():
            if hasattr(mc, k):
                assert getattr(mc, k) == v, k
        assert mc.physics.surface_forcing.scheme == "none"

    def test_overrides_win(self):
        mc = bench.bench_model_config(bench.BenchConfig(), A_h=123.0)
        assert mc.lateral_viscosity.A_h == 123.0


# ---------------------------------------------------------------------------
# Stability validator — planted-violation controls
# ---------------------------------------------------------------------------


def _state_with(**over):
    """Minimal duck-typed state carrying the validator's fields."""

    class _F:
        def __init__(self, data):
            self.data = data

    class _St:
        pass

    st = _St()
    nlat, nlon, nlev = 4, 8, 3
    shape3 = (nlat, nlon, nlev)
    st.eta = _F(jnp.zeros((nlat, nlon)))
    st.T = _F(jnp.full(shape3, 10.0))
    st.S = _F(jnp.full(shape3, 34.0))
    st.u = _F(jnp.zeros((nlat, nlon + 1, nlev)))
    st.v = _F(jnp.zeros((nlat + 1, nlon, nlev)))
    for k, val in over.items():
        setattr(st, k, _F(val))
    return st


class TestValidateGates:
    def test_clean_state_passes(self):
        ok, notes = bench.validate_results(_state_with(), {"max_speed": [0.01], "max_eta": [0.05]})
        assert ok

    def test_planted_nan_fails(self):
        # Control: the finiteness gate must FIRE on NaN.
        bad = _state_with(T=jnp.full((4, 8, 3), jnp.nan))
        ok, _ = bench.validate_results(bad, {})
        assert not ok

    def test_planted_overspeed_fails(self):
        # Control: the speed gate must FIRE on |u| > max_speed_limit.
        ok, _ = bench.validate_results(_state_with(), {"max_speed": [5.0], "max_eta": [0.05]})
        assert not ok

    def test_planted_overshot_eta_fails(self):
        # Control: the eta gate must FIRE on |eta| > max_eta_limit.
        ok, _ = bench.validate_results(_state_with(), {"max_speed": [0.01], "max_eta": [5.0]})
        assert not ok


# ---------------------------------------------------------------------------
# Registry / presets
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered(self):
        from legoesm.ocean.experiments import AVAILABLE_EXPERIMENTS

        assert "bench" in AVAILABLE_EXPERIMENTS
        assert AVAILABLE_EXPERIMENTS["bench"]["name"] == "bench"

    def test_recipe_tag(self):
        from legoesm.ocean.experiments.recipe_map import recipe_for

        assert recipe_for("bench") == "legoesm_nemo_like_v1"

    def test_presets_nemo_sizes(self):
        # NEMO BENCH namelist nn_isize/jsize: orca1 360x331, orca025
        # 1440x1206, orca12 4320x3146 — the latlon analog keeps n_lon.
        assert bench.BENCH_PRESETS["orca1_like"]["n_lon"] == 360
        assert bench.BENCH_PRESETS["orca12_like"]["n_lon"] == 4320
        for p in bench.BENCH_PRESETS.values():
            assert p["n_levels"] == 75  # nn_ksize = 75
            assert p["dt_seconds"] > 0
