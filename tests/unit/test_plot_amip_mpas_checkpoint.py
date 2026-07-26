"""Direct test for the AMIP MPAS checkpoint snapshot plotter.

The load-bearing pieces are the two that fail SILENTLY into a plausible but
wrong figure: picking the icosahedral subdivision level from the cell count
(a wrong level pairs every field with the wrong cell centre) and the
equal-area zonal binning (a wrong bin map smears the jet).
"""

import numpy as np
import pytest

from scripts.plot.plot_amip_mpas_checkpoint import (
    latest_checkpoint,
    load_state,
    main,
    subdivisions_for,
    zonal_mean,
)


def _checkpoint(path, n_cells=42, nlev=4):
    """A minimal checkpoint with the keys the plotter reads."""
    half = np.linspace(0.01, 1.0, nlev + 1)
    np.savez(path,
             T=np.full((n_cells, nlev), 250.0),
             trc_q_v=np.full((n_cells, nlev), 1e-3),
             u=np.zeros((n_cells * 3, nlev)),
             p_s=np.full(n_cells, 1e5),
             meta_vgrid=np.stack([np.zeros(nlev + 1), half]))


class TestSubdivisions:
    @pytest.mark.parametrize("k,n", [(0, 12), (1, 42), (3, 642), (5, 10242)])
    def test_known_icosahedral_counts(self, k, n):
        assert subdivisions_for(n) == k

    def test_non_icosahedral_count_raises(self):
        """Never guess: a silently-wrong mesh renders a fictitious map."""
        with pytest.raises(SystemExit, match="icosahedral"):
            subdivisions_for(10000)


class TestLatestCheckpoint:
    def test_picks_highest_day(self, tmp_path):
        for d in (30, 200, 90):
            _checkpoint(tmp_path / f"checkpoint_day_{d:04d}.npz")
        assert latest_checkpoint(tmp_path).name == "checkpoint_day_0200.npz"

    def test_explicit_day(self, tmp_path):
        for d in (30, 200):
            _checkpoint(tmp_path / f"checkpoint_day_{d:04d}.npz")
        assert latest_checkpoint(tmp_path, 30).name == "checkpoint_day_0030.npz"

    def test_missing_day_raises_listing_what_exists(self, tmp_path):
        _checkpoint(tmp_path / "checkpoint_day_0030.npz")
        with pytest.raises(SystemExit, match=r"\[30\]"):
            latest_checkpoint(tmp_path, 999)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="no checkpoint"):
            latest_checkpoint(tmp_path)


class TestLoadState:
    def test_shapes_and_dsigma(self, tmp_path):
        p = tmp_path / "checkpoint_day_0005.npz"
        _checkpoint(p, n_cells=42, nlev=4)
        T, q_v, u, p_s, dsigma = load_state(p)  # noqa: N806 (T = temperature)
        assert T.shape == (42, 4) and q_v.shape == (42, 4)
        assert p_s.shape == (42,)
        assert dsigma.shape == (4,)
        # dsigma spans the full column: sum = sigma_top..1
        np.testing.assert_allclose(dsigma.sum(), 0.99, atol=1e-12)


class TestZonalMean:
    def test_bins_are_equal_area(self):
        """Equal-area bins are uniform in sin(lat), so they widen toward the
        poles — a uniform-in-latitude binning would leave polar bins nearly
        empty on an SCVT mesh."""
        lat = np.linspace(-89.0, 89.0, 4000)
        field = np.ones((4000, 2))
        lats, prof = zonal_mean(field, lat, n_bins=8)
        widths = np.diff(np.sort(lats))
        assert widths[0] > widths[len(widths) // 2], (
            "bins should widen away from the equator")
        assert prof.shape[1] == 2

    def test_mean_recovers_a_latitude_ramp(self):
        lat = np.linspace(-89.0, 89.0, 2000)
        field = lat[:, None] * np.ones((1, 3))
        lats, prof = zonal_mean(field, lat, n_bins=10)
        # each bin's mean tracks its centre latitude
        np.testing.assert_allclose(prof[:, 0], lats, atol=6.0)

    def test_empty_bins_dropped_not_nan(self):
        """All cells in one hemisphere: the empty bins must vanish rather than
        emit NaN rows that would blank the contour plot."""
        lat = np.linspace(10.0, 80.0, 500)
        lats, prof = zonal_mean(np.ones((500, 2)), lat, n_bins=12)
        assert np.isfinite(prof).all()
        assert len(lats) == prof.shape[0] < 12


def test_end_to_end_writes_a_figure(tmp_path):
    """Exercises the real mesh + shared CWV/velocity helpers at the smallest
    icosahedral size."""
    pytest.importorskip("matplotlib")
    _checkpoint(tmp_path / "checkpoint_day_0007.npz", n_cells=42, nlev=4)
    assert main([str(tmp_path)]) == 0
    assert (tmp_path / "snapshot_day_0007.png").exists()


def test_missing_run_dir_raises():
    with pytest.raises(SystemExit, match="no such run directory"):
        main(["/nonexistent/run/dir"])
