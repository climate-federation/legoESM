import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import plot_phi_panels as pp  # noqa: E402


def test_per_site_bins_even_days_by_side_and_count():
    n = pp.MIN_PER_BIN
    z = np.r_[np.full(n, 0.15), np.full(n - 1, 1.5), np.full(n, -0.15)]
    ph = np.r_[np.full(n, 2.0), np.full(n - 1, 5.0), np.full(n, 0.7)]
    odd = (np.full(n, 0.15), np.full(n, 9.0))
    data = {"s": {"m": {0: (z, ph), 1: odd}}}
    stable = pp.per_site(data, "m", 1)[0]
    unstable = pp.per_site(data, "m", -1)[0]
    i = np.searchsorted(pp.EDGES, 0.15) - 1
    assert stable[i] == 2.0 and unstable[i] == 0.7
    assert np.isnan(stable[np.searchsorted(pp.EDGES, 1.5) - 1])  # one sample short
    assert np.isfinite(stable).sum() == 1 and np.isfinite(unstable).sum() == 1
