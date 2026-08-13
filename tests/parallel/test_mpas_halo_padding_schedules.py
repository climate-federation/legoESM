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
    _cost, _greedy_first_fit, _pair_extents, _size_banded,
)

CW, EW = 4, 1     # 1 cell row = 4 bytes, 1 edge row = 1 byte, in tests


def _matching_ok(rounds):
    for rnd in rounds:
        seen = set()
        for a, b in rnd:
            if a in seen or b in seen:
                return False
            seen.add(a)
            seen.add(b)
    return True


def test_cost_matches_production_padding_semantics():
    """A round pads CELLS and EDGES independently, and BOTH endpoints of
    every pair send the padded buffer — so the round costs
    2 * n_pairs * (max_cells * cell_w + max_edges * edge_w). Charging
    the max of the byte SUM, or one send per pair, both understate it."""
    ext = {(0, 1): (1, 30), (2, 3): (3, 1)}
    # max_c = 3, max_e = 30 -> per-buffer 3*4 + 30*1 = 42; 2 pairs, both
    # endpoints send -> 4 * 42
    assert _cost([[(0, 1), (2, 3)]], ext, CW, EW) == 4 * 42
    # split into two rounds and each pays only its own extents
    assert _cost([[(0, 1)], [(2, 3)]], ext, CW, EW) == \
        2 * (1 * 4 + 30 * 1) + 2 * (3 * 4 + 1 * 1)


def test_cost_pads_cells_and_edges_separately():
    """A pair that is fat in CELLS must not be priced as if it were also
    fat in EDGES: the two extents are independent maxima."""
    ext = {(0, 1): (100, 1), (2, 3): (1, 100)}
    together = _cost([[(0, 1), (2, 3)]], ext, CW, EW)
    apart = _cost([[(0, 1)], [(2, 3)]], ext, CW, EW)
    assert together == 4 * (100 * 4 + 100 * 1)
    assert apart < together


def test_pair_extents_are_undirected_and_take_the_larger_direction():
    comm = [(0, 1), (1, 0), (2, 3)]
    cell_send = {(0, 1): [0, 1, 2], (1, 0): [0], (2, 3): [0, 1]}
    edge_send = {(0, 1): [0], (1, 0): [0, 1, 2, 3], (2, 3): []}
    ext = _pair_extents(comm, cell_send, edge_send)
    assert set(ext) == {(0, 1), (2, 3)}, "directions must collapse"
    assert ext[(0, 1)] == (3, 4)
    assert ext[(2, 3)] == (2, 0)


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
    ext = {p: (sizes[p], 0) for p in pairs}
    assert _cost(banded, ext, CW, EW) < _cost(flat, ext, CW, EW)


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
