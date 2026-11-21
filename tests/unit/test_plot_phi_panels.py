import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import plot_phi_panels as pp  # noqa: E402
import standard_formulations as sf  # noqa: E402


def test_per_site_bins_even_days_by_side_and_count():
    assert pp.MIN_PER_BIN == 30  # same per-bin minimum as plot_round2.site_bins
    n = 30
    z = np.r_[np.full(n, 0.15), np.full(n - 1, 1.5), np.full(n, -0.15)]
    ph = np.r_[np.full(n, 2.0), np.full(n - 1, 5.0), np.full(n, 0.7)]
    odd = (np.full(n, 0.15), np.full(n, 9.0))
    data = {"s": {"m": {0: (z, ph), 1: odd}}}
    stable, zs = pp.per_site(data, "m", 1)
    unstable, zu = pp.per_site(data, "m", -1)
    i = np.searchsorted(pp.EDGES, 0.15) - 1
    assert stable[0, i] == 2.0 and unstable[0, i] == 0.7
    assert zs[0, i] == 0.15 and zu[0, i] == 0.15
    assert np.isnan(stable[0, np.searchsorted(pp.EDGES, 1.5) - 1])  # one sample short
    assert np.isfinite(stable).sum() == 1 and np.isfinite(unstable).sum() == 1


def test_ratio_uses_each_samples_own_zeta():
    # three groups in bin [1, 2) with per-sample ratios 1, 2, 1.2: true median 1.2; median(phi)/f(median zeta) gives
    # 12.6/8.5 and a bin-wide zeta gives about 1.51, so neither shortcut passes
    z = np.repeat([1.0, 1.5, 1.9], 10)
    f = lambda x: 1 + 5 * x  # noqa: E731
    ph = f(z) * np.repeat([1.0, 2.0, 1.2], 10)
    data = {"s": {"m": {0: (z, ph), 1: (z, ph)}}}
    ratio, _ = pp.per_site(data, "m", 1, f)
    i = np.searchsorted(pp.EDGES, 1.5) - 1
    assert abs(ratio[0, i] - 1.2) < 1e-12
    assert abs(np.median(ph) / f(np.median(z)) - 1.2) > 0.1


def test_scheme_phi_in_tower_kappa_convention():
    zeta = np.array([0.3, 2.0])
    s = sf.BUSINGER_KAPPA / pp.KAPPA
    assert np.allclose(pp.scheme_phi("businger1971", "m", zeta), sf.phi("businger1971", "m", zeta * s) / s)
    assert not np.allclose(pp.scheme_phi("businger1971", "m", zeta), sf.phi("businger1971", "m", zeta))
    assert np.allclose(pp.scheme_phi("dyer1974", "m", zeta), sf.phi("dyer1974", "m", zeta))


def test_ratio_refuses_nonpositive_prediction():
    import pytest
    z = np.full(30, 1.5)
    data = {"s": {"m": {0: (z, z), 1: (z, z)}}}
    with pytest.raises(ValueError):
        pp.per_site(data, "m", 1, lambda x: 1 - x)
