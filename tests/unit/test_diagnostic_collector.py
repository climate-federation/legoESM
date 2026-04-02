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
