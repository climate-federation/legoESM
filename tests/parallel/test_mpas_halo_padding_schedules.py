"""Direct test for the offline halo-padding schedule explorer.

The script reports how much of the MPAS halo payload is padding under
different colourings. Its two claims have to hold or every inflation
number it prints is wrong:

* a "round" really is a MATCHING — no device may appear twice in one
  round, or the schedule it prices could never be executed;
* the padded cost really is ``n_pairs * round_max`` summed over rounds,
  which is what ``ppermute`` shipping a fixed-shape buffer costs.

Both are checked on a hand-built graph whose answer can be computed by
hand, so a regression in the packing heuristics cannot hide behind a
plausible-looking number on the real mesh.
"""

import pytest

from scripts.validate.mpas_halo_padding_schedules import (
    _cost, _greedy_first_fit, _size_banded,
)


def _matching_ok(rounds):
    for rnd in rounds:
        seen = set()
        for a, b in rnd:
            if a in seen or b in seen:
                return False
            seen.add(a)
            seen.add(b)
    return True


def test_cost_is_pairs_times_round_max():
    # one round of three pairs sized 10, 20, 30 costs 3 * 30, not 60
    sizes = {(0, 1): 10, (2, 3): 20, (4, 5): 30}
    assert _cost([[(0, 1), (2, 3), (4, 5)]], sizes) == 90
    # split so each round is size-homogeneous and the cost drops to the
    # true total
    assert _cost([[(0, 1)], [(2, 3)], [(4, 5)]], sizes) == 60


def test_first_fit_produces_a_proper_matching_per_round():
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    sizes = {p: 1 for p in pairs}
    rounds = _greedy_first_fit(pairs, sizes, pairs)
    assert _matching_ok(rounds)
    assert sum(len(r) for r in rounds) == len(pairs)


def test_size_banding_never_costs_more_than_one_band():
    """Banding exists to stop one fat pair setting a whole round's
    price; on a deliberately skewed graph it must beat the unbanded
    first fit, or the heuristic is pointless."""
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    sizes = {(0, 1): 1000, (2, 3): 10, (4, 5): 10, (6, 7): 10}
    flat = _greedy_first_fit(pairs, sizes, pairs)
    banded = _size_banded(pairs, sizes, 2)
    assert _matching_ok(banded)
    assert _cost(banded, sizes) < _cost(flat, sizes)


def test_size_banding_keeps_every_pair():
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    sizes = {p: 10 * (i + 1) for i, p in enumerate(pairs)}
    for nb in (2, 3, 5):
        rounds = _size_banded(pairs, sizes, nb)
        assert _matching_ok(rounds)
        flat = [p for r in rounds for p in r]
        assert sorted(flat) == sorted(pairs), (
            f"{nb} bands dropped or duplicated a pair")


@pytest.mark.parametrize("nb", [1, 2, 4])
def test_banding_degenerates_gracefully_on_uniform_sizes(nb):
    pairs = [(0, 1), (2, 3), (4, 5)]
    sizes = {p: 7 for p in pairs}
    rounds = _size_banded(pairs, sizes, nb)
    assert _matching_ok(rounds)
    assert sorted(p for r in rounds for p in r) == sorted(pairs)
