"""Unit tests for DiagnosticCollector snapshot serialization."""

import numpy as np
import pytest

from legoesm.driver.diagnostics import DiagnosticCollector


@pytest.fixture()
def collector():
    """Create a minimal DiagnosticCollector."""
    nlev = 5
    sigma_full = np.linspace(0.1, 0.9, nlev)
    dsigma = np.diff(np.linspace(0, 1, nlev + 1))
    return DiagnosticCollector(
        nlev=nlev,
        sigma_full=sigma_full,
        dsigma=dsigma,
        n_days=30,
    )


class TestSnapshotSerialization:
    """Tests for snapshot capture and save to disk."""

    def test_snapshot_days_populated(self, collector):
        """Snapshot days are populated based on n_days."""
        assert 5 in collector.snapshot_days
        assert 10 in collector.snapshot_days
        assert 30 in collector.snapshot_days  # n_days always included
        assert 200 not in collector.snapshot_days  # > n_days excluded

    def test_snapshots_dict_initially_empty(self, collector):
        assert collector.snapshots == {}

    def test_save_with_snapshots(self, collector, tmp_path):
        """Snapshots dict is correctly serialized to snapshots.npz."""
        # Manually add fake snapshot data
        collector.snapshots[5] = {
            "SST": np.ones((6, 4, 4)),
            "T_low": np.ones((6, 4, 4)) * 280.0,
        }
        collector.snapshots[10] = {
            "SST": np.ones((6, 4, 4)) * 290.0,
            "T_low": np.ones((6, 4, 4)) * 275.0,
        }

        # Add minimal timeseries data so save() doesn't fail
        collector.times.append(5.0)
        collector.sst.append(290.0)
        collector.sic.append(0.0)
        collector.T_atm.append(250.0)
        collector.T_low.append(280.0)
        collector.max_wind.append(30.0)
        collector.precip.append(3.0)
        collector.CWV.append(25.0)
        collector.sw_up_toa.append(100.0)
        collector.lw_up_toa.append(240.0)
        collector.sw_net_sfc.append(180.0)
        collector.lw_net_sfc.append(-50.0)
        collector.dry_mass.append(101325.0)

        collector.save(tmp_path)

        # Check snapshots.npz was created
        snap_path = tmp_path / "snapshots.npz"
        assert snap_path.exists()

        snap_data = dict(np.load(snap_path))
        assert "snapshot_days" in snap_data
        np.testing.assert_array_equal(snap_data["snapshot_days"], [5, 10])
        assert "day005_SST" in snap_data
        assert "day010_SST" in snap_data
        assert "day005_T_low" in snap_data
        np.testing.assert_allclose(snap_data["day005_SST"], 1.0)
        np.testing.assert_allclose(snap_data["day010_SST"], 290.0)

    def test_save_without_snapshots(self, collector, tmp_path):
        """Save works when no snapshots were captured."""
        collector.times.append(1.0)
        collector.sst.append(290.0)
        collector.sic.append(0.0)
        collector.T_atm.append(250.0)
        collector.T_low.append(280.0)
        collector.max_wind.append(30.0)
        collector.precip.append(3.0)
        collector.CWV.append(25.0)
        collector.sw_up_toa.append(100.0)
        collector.lw_up_toa.append(240.0)
        collector.sw_net_sfc.append(180.0)
        collector.lw_net_sfc.append(-50.0)
        collector.dry_mass.append(101325.0)

        collector.save(tmp_path)

        # timeseries.npz should exist, but no snapshots.npz
        assert (tmp_path / "timeseries.npz").exists()
        assert not (tmp_path / "snapshots.npz").exists()


class TestFlushToDisk:
    """Tests for incremental flush."""

    def test_flush_creates_chunk(self, collector, tmp_path):
        """flush_to_disk creates numbered chunk files."""
        collector.times.extend([1.0, 2.0])
        collector.T_atm.extend([250.0, 251.0])
        collector.T_low.extend([280.0, 281.0])
        collector.max_wind.extend([30.0, 31.0])
        collector.precip.extend([3.0, 3.1])
        collector.CWV.extend([25.0, 25.1])
        collector.sw_up_toa.extend([100.0, 100.0])
        collector.lw_up_toa.extend([240.0, 240.0])
        collector.sw_net_sfc.extend([180.0, 180.0])
        collector.lw_net_sfc.extend([-50.0, -50.0])
        collector.dry_mass.extend([101325.0, 101325.0])
        collector.sst.extend([290.0, 290.0])
        collector.sic.extend([0.0, 0.0])

        collector.flush_to_disk(tmp_path)

        incr_dir = tmp_path / "timeseries_incremental"
        assert incr_dir.exists()
        chunks = sorted(incr_dir.glob("chunk_*.npz"))
        assert len(chunks) == 1

        # Lists should be cleared after flush
        assert len(collector.times) == 0
        assert len(collector.T_atm) == 0


class TestCollectTLowMean:
    """CMOR ``tas`` uses the segment-mean lowest-level T when provided.

    ``collect(..., t_low_mean=...)`` must shift the spatial-monthly ``tas``
    field by exactly ``t_low_mean - T_low_instantaneous`` relative to a
    collect without it (the instantaneous MOST 2 m offset is common to
    both).  Guards the CMOR diurnal-alias fix at the collector level.
    """

    _NLAT, _NLON, _NLEV = 36, 72, 5

    def _make_collector(self):
        import jax.numpy as jnp

        nlev = self._NLEV
        # jnp (not np): collect() feeds these through the energy tracker's
        # lax.scan, which indexes them with a traced level index.
        sigma_full = jnp.linspace(0.1, 0.9, nlev)
        dsigma = jnp.diff(jnp.linspace(0, 1, nlev + 1))
        coll = DiagnosticCollector(
            nlev=nlev,
            sigma_full=sigma_full,
            dsigma=dsigma,
            monthly_means=True,
            cmip_output=True,
            n_days=30,
            cmip_resolution_deg=5.0,   # 36 x 72 == native -> identity regrid
        )

        class _Grid:
            lat = np.deg2rad(np.linspace(-87.5, 87.5, 36))
            lon = np.deg2rad(np.linspace(2.5, 357.5, 72))

        coll.set_cmip_grid_info("latlon", grid=_Grid(), start_year=1979)
        return coll

    def _collect(self, coll, t_low_mean=None, **flux_kw):
        import jax.numpy as jnp
        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState

        s3 = (self._NLAT, self._NLON, self._NLEV)
        s2 = (self._NLAT, self._NLON)
        T = jnp.full(s3, 280.0)
        state = HydrostaticState(
            u=Field(jnp.zeros(s3), name="u", dims=("lat", "lon", "level"), units="m/s"),
            v=Field(jnp.zeros(s3), name="v", dims=("lat", "lon", "level"), units="m/s"),
            T=Field(T, name="T", dims=("lat", "lon", "level"), units="K"),
            p_s=Field(jnp.full(s2, 101325.0), name="p_s", dims=("lat", "lon"), units="Pa"),
            phis=Field(jnp.zeros(s2), name="phis", dims=("lat", "lon"), units="m2/s2"),
        )
        coll.collect(
            elapsed_day=5.0,
            day=5.0,
            state=state,
            q_v=jnp.full(s3, 0.005),
            q_c=jnp.zeros(s3),
            q_r=jnp.zeros(s3),
            sst=jnp.full(s2, 290.0),
            sic=jnp.zeros(s2),
            precip_total=jnp.zeros(s2),
            sw_up_toa=jnp.full(s2, 100.0),
            lw_up_toa=jnp.full(s2, 240.0),
            sw_net_sfc=jnp.full(s2, 160.0),
            lw_net_sfc=jnp.full(s2, -60.0),
            sw_down_toa=jnp.full(s2, 340.0),
            T_ice=271.35,
            t_low_mean=t_low_mean,
            **flux_kw,
        )

    def test_latent_heat_without_water_is_refused(self):
        """hfls fed alone must not be turned into evspsbl = hfls / L_v."""
        import jax.numpy as jnp
        coll = self._make_collector()
        s2 = (self._NLAT, self._NLON)
        with pytest.raises(ValueError, match="lhflx was fed without evspsbl"):
            self._collect(coll, lhflx=jnp.full(s2, 80.0))

    def test_fed_water_is_the_series_and_the_field(self):
        """The evspsbl series, monthly field and moisture closure all carry
        the FED water, not a latent-heat inverse."""
        import jax.numpy as jnp

        from legoesm import constants
        coll = self._make_collector()
        s2 = (self._NLAT, self._NLON)
        e = 3.0e-5
        self._collect(coll, lhflx=jnp.full(s2, 80.0), evspsbl=jnp.full(s2, e))
        assert coll.evspsbl[-1] == pytest.approx(e, rel=1e-6)
        assert abs(coll.evspsbl[-1] / (80.0 / constants.L_v) - 1.0) > 0.05
        (bucket,) = coll._spatial_monthly._data_2d.values()
        arr, count = bucket["evspsbl"]
        assert count == 1
        np.testing.assert_allclose(np.asarray(arr), e, rtol=1e-6)
        assert coll.moisture_tracker.evap_rate[-1] == pytest.approx(
            e * 86400.0, rel=1e-6)

    def test_lightweight_path_refuses_heat_without_water_before_recording(self):
        """collect_lightweight raises on lhflx-without-evspsbl BEFORE any
        series grows: no NaN evspsbl beside a real hfls, ever."""
        import jax.numpy as jnp
        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState
        coll = self._make_collector()
        s3 = (self._NLAT, self._NLON, self._NLEV)
        s2 = (self._NLAT, self._NLON)
        state = HydrostaticState(
            u=Field(jnp.zeros(s3), name="u", dims=("lat", "lon", "level"), units="m/s"),
            v=Field(jnp.zeros(s3), name="v", dims=("lat", "lon", "level"), units="m/s"),
            T=Field(jnp.full(s3, 280.0), name="T", dims=("lat", "lon", "level"), units="K"),
            p_s=Field(jnp.full(s2, 101325.0), name="p_s", dims=("lat", "lon"), units="Pa"),
            phis=Field(jnp.zeros(s2), name="phis", dims=("lat", "lon"), units="m2/s2"),
        )
        with pytest.raises(ValueError, match="collect_lightweight: lhflx was fed without evspsbl"):
            coll.collect_lightweight(
                5.0, state, jnp.full(s3, 0.005), jnp.full(s2, 290.0),
                jnp.zeros(s2), jnp.zeros(s2), jnp.full(s2, 100.0),
                jnp.full(s2, 240.0), jnp.full(s2, 160.0), jnp.full(s2, -60.0),
                lhflx=jnp.full(s2, 80.0))
        assert coll.times == [] and coll.hfls == [] and coll.evspsbl == []

    def _monthly_tas_sum(self, coll):
        (bucket,) = coll._spatial_monthly._data_2d.values()
        arr, count = bucket["tas"]
        assert count == 1
        return np.asarray(arr)

    def test_t_low_mean_shifts_cmor_tas(self):
        import jax.numpy as jnp

        coll_inst = self._make_collector()
        self._collect(coll_inst, t_low_mean=None)
        tas_inst = self._monthly_tas_sum(coll_inst)

        offset = 5.0
        coll_mean = self._make_collector()
        self._collect(
            coll_mean,
            t_low_mean=jnp.full((self._NLAT, self._NLON), 280.0 + offset),
        )
        tas_mean = self._monthly_tas_sum(coll_mean)

        np.testing.assert_allclose(tas_mean - tas_inst, offset, rtol=1e-6)


class _Field:
    def __init__(self, arr):
        import jax.numpy as _jnp
        self.data = _jnp.asarray(arr, dtype=_jnp.float64)


class _FakeState:
    """Duck-typed state exposing just the fields check_stability reads."""
    def __init__(self, u, T, p_s):
        self.u = _Field(u)
        self.T = _Field(T)
        self.p_s = _Field(p_s)


def test_check_stability_flags_nonfinite_p_s(collector):
    """A NaN surface pressure with finite, in-bounds winds/T must be caught.
    Before the fix, min/max p_s were NaN and both ``NaN < lo`` and ``NaN > hi``
    were False, so the state passed the blow-up probe and could be checkpointed
    at a wallclock-graceful exit — the chain would then restart from garbage."""
    healthy = _FakeState(
        u=np.full((4, 8), 5.0),        # finite, calm
        T=np.full((4, 8), 288.0),      # finite, in-bounds
        p_s=np.full((6,), 1.0e5),      # healthy
    )
    assert collector.check_stability(healthy, 5.0) is None

    nan_ps = _FakeState(
        u=np.full((4, 8), 5.0),
        T=np.full((4, 8), 288.0),
        p_s=np.array([1.0e5, np.nan, 1.0e5, 1.0e5, 1.0e5, 1.0e5]),
    )
    reason = collector.check_stability(nan_ps, 5.0)
    assert reason is not None and "surface pressure" in reason


def test_check_stability_flags_nonfinite_v(collector):
    """A NaN in the meridional wind v must be caught. wind_term =
    max(sqrt(u^2+v^2)) is NaN, and NaN > 500 is False, so a v-only blow-up would
    pass the max-wind check unless v's finiteness is checked explicitly."""
    class _StateUV:
        def __init__(self, u, v, T, p_s):
            self.u = _Field(u)
            self.v = _Field(v)
            self.T = _Field(T)
            self.p_s = _Field(p_s)

    v = np.full((4, 8), 3.0)
    v[1, 1] = np.nan
    st = _StateUV(
        u=np.full((4, 8), 5.0), v=v,
        T=np.full((4, 8), 288.0), p_s=np.full((6,), 1.0e5),
    )
    reason = collector.check_stability(st, 5.0)
    assert reason is not None and "wind" in reason


class TestCmipLonAlignment:
    """CMOR regrid output must match the [0, 360) lon labels.

    The cube/Voronoi regridders emit columns on lon [-180, 180) while
    ``_cmip_target_latlon`` labels the file [0, 360); the missing
    half-turn roll shifted every cubed-sphere CMOR map by 180 deg
    (tuned-year clt vs coastlines, 2026-07-21).  Analytic tripwire:
    regrid sin(lon_cell) off a real cube grid and require it to land
    under the matching sin(lon_label) columns — plus the synthetic
    violation (un-rolling it back) must fail loudly.
    """

    def _cube_collector_and_field(self):
        import jax.numpy as jnp
        from legoesm.grids.cubed_sphere import create_cubed_sphere

        nlev = 3
        coll = DiagnosticCollector(
            nlev=nlev,
            sigma_full=jnp.linspace(0.1, 0.9, nlev),
            dsigma=jnp.diff(jnp.linspace(0, 1, nlev + 1)),
            monthly_means=True,
            cmip_output=True,
            n_days=30,
            cmip_resolution_deg=5.0,  # 36 x 72 target
        )
        grid = create_cubed_sphere(24)
        coll.set_cmip_grid_info("cubed_sphere", grid=grid, start_year=1979)
        field = np.sin(np.asarray(grid.lon))  # (6, n, n), lon in rad
        return coll, field

    def test_cube_regrid_matches_lon_labels(self):
        coll, field = self._cube_collector_and_field()
        out = coll._regrid_to_latlon_2d(field)
        assert out is not None and out.shape == (36, 72)
        _, lon_lab = coll._cmip_target_latlon()
        expect = np.broadcast_to(
            np.sin(np.deg2rad(lon_lab))[None, :], out.shape)
        # Away from the poles bilinear-off-C24 is accurate to a few %.
        band = slice(4, 32)
        err_fixed = np.max(np.abs(out[band] - expect[band]))
        assert err_fixed < 0.1, f"regridded sin(lon) misaligned: {err_fixed}"
        # Synthetic violation: undo the roll -> half-turn shift -> sign flip.
        shifted = np.roll(out, -out.shape[1] // 2, axis=1)
        err_shifted = np.max(np.abs(shifted[band] - expect[band]))
        assert err_shifted > 1.5, "tripwire vacuous: shifted field also fits"

    def test_cube_regrid_3d_matches_lon_labels(self):
        coll, field = self._cube_collector_and_field()
        f3 = np.repeat(field[..., None], 3, axis=-1)
        out = coll._regrid_to_latlon_3d(f3)
        assert out is not None and out.shape == (36, 72, 3)
        _, lon_lab = coll._cmip_target_latlon()
        expect = np.sin(np.deg2rad(lon_lab))[None, :, None]
        err = np.max(np.abs(out[4:32] - np.broadcast_to(
            expect, out.shape)[4:32]))
        assert err < 0.1

    def test_odd_nlon_raises(self):
        import jax.numpy as jnp
        coll = DiagnosticCollector(
            nlev=3,
            sigma_full=jnp.linspace(0.1, 0.9, 3),
            dsigma=jnp.diff(jnp.linspace(0, 1, 4)),
            monthly_means=True,
            cmip_output=True,
            n_days=30,
            cmip_resolution_deg=5.0,
        )
        coll._cmip_nlon = 71  # force odd
        with pytest.raises(ValueError, match="even"):
            coll._roll_to_cmip_lon(np.zeros((36, 71)))
