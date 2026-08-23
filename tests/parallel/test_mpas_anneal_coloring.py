"""Gates for the annealed halo round schedule.

Every pair in a coloured halo round ships that round's largest halo, because
the sharded step needs one shape across devices. At 64 devices that padding is
34% of the rows on the wire; at 128 devices it is 69%. An annealed search over
the same move set the model already uses -- give one exchange a different
round -- takes those to 26% and 39%.

The schedule decides who sends what to whom, so a wrong one does not fail
loudly: it delivers the wrong rows. These gates say the schedule stays legal,
stays reproducible across processes, and does not move the answer.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy


def _need(n: int) -> None:
    if jax.device_count() < n:
        pytest.skip(
            f"Need {n} CPU devices "
            f"(XLA_FLAGS=--xla_force_host_platform_device_count={n})")


def _ring(n_ranks: int):
    """A communication graph with a known chromatic number and lopsided
    payloads: every rank talks to its two neighbours, and one exchange is a
    hundred times heavier than the rest. A descent that only moves downhill
    leaves the heavy pair where its seed put it."""
    pairs = {tuple(sorted((i, (i + 1) % n_ranks))) for i in range(n_ranks)}
    weights = {p: (1, 1) for p in pairs}
    heavy = sorted(pairs)[0]
    weights[heavy] = (100, 100)
    return sorted(pairs), weights


# ---------------------------------------------------------------- resolver


@pytest.mark.parametrize("value,expected", [("", 0), ("0", 0),
                                            ("2", 2_000_000),
                                            ("40", 40_000_000)])
def test_the_switch_reads_millions_of_moves(value, expected):
    from legoesm.parallel.sharded_dynamics import _resolve_anneal_coloring

    assert _resolve_anneal_coloring(value) == expected


@pytest.mark.parametrize("value", ["yes", "-1", "2.5", "01", "1e6", " 2"])
def test_a_typo_does_not_silently_pick_a_schedule(value):
    """A misspelt budget must not quietly leave the search off: the run would
    ship a different schedule from the one its receipt claims."""
    from legoesm.parallel.sharded_dynamics import _resolve_anneal_coloring

    with pytest.raises(ValueError):
        _resolve_anneal_coloring(value)


def test_an_unreachable_budget_is_refused():
    from legoesm.parallel.sharded_dynamics import _resolve_anneal_coloring

    with pytest.raises(ValueError, match="out of range"):
        _resolve_anneal_coloring("500")


# ---------------------------------------------------------------- the search


def test_the_search_returns_a_legal_schedule():
    """Two exchanges sharing a rank in one round would collide on the device.
    This is the property that makes the schedule usable at all."""
    from legoesm.parallel.sharded_dynamics import (
        anneal_coloring, check_proper_edge_coloring, greedy_edge_coloring,
    )

    pairs, weights = _ring(16)
    seed = greedy_edge_coloring(set(pairs))
    out = anneal_coloring(set(pairs), weights, seed,
                          max(seed.values()) + 2, 200_000)
    assert check_proper_edge_coloring(out, set(pairs))
    assert set(out) == set(pairs), "the search dropped or invented an exchange"


def test_the_search_is_reproducible():
    """Every process of a multi-controller run derives its own schedule and
    they must agree exactly, so the search may not depend on a clock, on
    dictionary order, or on an unseeded generator."""
    from legoesm.parallel.sharded_dynamics import (
        anneal_coloring, greedy_edge_coloring,
    )

    pairs, weights = _ring(24)
    seed = greedy_edge_coloring(set(pairs))
    kwargs = dict(n_colors=max(seed.values()) + 2, n_moves=200_000)
    first = anneal_coloring(set(pairs), weights, seed, **kwargs)
    second = anneal_coloring(set(pairs), weights, dict(seed), **kwargs)
    assert first == second


def test_the_search_actually_lowers_the_wire_weight():
    """Non-vacuity: on a graph built so the greedy seed is bad, the search has
    to beat it. Without this the gate above passes on a search that returns
    its input."""
    from legoesm.parallel.sharded_dynamics import (
        anneal_coloring, greedy_edge_coloring, padded_weight,
    )

    pairs, weights = _ring(16)
    seed = greedy_edge_coloring(set(pairs))
    before = padded_weight(seed, weights)
    out = anneal_coloring(set(pairs), weights, seed,
                          max(seed.values()) + 2, 400_000)
    assert padded_weight(out, weights) < before, (
        f"the search did not improve on its seed: {before} -> "
        f"{padded_weight(out, weights)}")


def test_more_moves_never_return_a_worse_schedule():
    """The search keeps the best colouring it has seen, so spending more of
    the budget cannot lose ground."""
    from legoesm.parallel.sharded_dynamics import (
        anneal_coloring, greedy_edge_coloring, padded_weight,
    )

    pairs, weights = _ring(20)
    seed = greedy_edge_coloring(set(pairs))
    n_colors = max(seed.values()) + 2
    short = padded_weight(
        anneal_coloring(set(pairs), weights, seed, n_colors, 50_000), weights)
    long = padded_weight(
        anneal_coloring(set(pairs), weights, seed, n_colors, 500_000), weights)
    assert long <= short


# ---------------------------------------------------------------- the model


def _build(devices: int, *, anneal: str, monkeypatch):
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.parallel.voronoi_partition import (
        reorder_voronoi_for_sharding,
    )
    from legoesm.parallel.mesh import (
        create_voronoi_device_mesh, replicate_pytree,
    )
    from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step

    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    monkeypatch.setenv("LEGOESM_MPAS_ANNEAL_COLORING", anneal)
    monkeypatch.setenv("LEGOESM_MPAS_WIDE_HALO", "1")
    monkeypatch.setenv("LEGOESM_MPAS_RAGGED_HALO", "0")
    monkeypatch.setenv("LEGOESM_MPAS_HALO_MERGE_SCATTER", "0")

    mesh = create_voronoi_mesh(subdivision_level=4)
    mesh = reorder_voronoi_for_sharding(mesh, devices)
    sigma = create_sigma_coordinate(8)
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
        pv_scheme="energy", time_integrator="ssp_rk3")
    dev_config = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges,
        nVertices=mesh.nVertices, n_devices=devices)
    model = MPASPrimitiveEquationModel(
        replicate_pytree(mesh, dev_config), sigma, cfg)
    state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
    step = make_voronoi_sharded_step(
        model, dev_config, halo_strategy="ppermute")
    return step, state, 600.0


def _assert_the_search_changes_this_mesh(devices: int) -> None:
    """Non-vacuity for the gate below: unless the search actually moves an
    exchange to a different round on THIS mesh, comparing the model's output
    with the switch on and off compares the same schedule twice and proves
    nothing (codex review)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.voronoi_partition import (
        reorder_voronoi_for_sharding,
    )
    from legoesm.parallel.sharded_dynamics import (
        anneal_coloring, greedy_edge_coloring, multi_ordering_edge_coloring,
        size_aware_edge_coloring,
    )

    mesh = reorder_voronoi_for_sharding(
        create_voronoi_mesh(subdivision_level=4), devices)
    cells_per = int(mesh.nCells) // devices
    owner = np.repeat(np.arange(devices), cells_per)
    cells_on_cell = np.asarray(mesh.cellsOnCell)
    pairs, weights = set(), {}
    for rank in range(devices):
        owned = np.arange(rank * cells_per, (rank + 1) * cells_per)
        nb = cells_on_cell[:, owned].ravel()
        nb = nb[(nb >= 0) & (nb < devices * cells_per)]
        for other, count in zip(*np.unique(owner[nb], return_counts=True)):
            if int(other) == rank:
                continue
            key = tuple(sorted((rank, int(other))))
            pairs.add(key)
            weights[key] = (max(weights.get(key, (0, 0))[0], int(count)),
                            max(weights.get(key, (0, 0))[1], int(count)))
    greedy = greedy_edge_coloring(pairs)
    multi, _ = multi_ordering_edge_coloring(pairs)
    seed = multi if max(multi.values()) < max(greedy.values()) else greedy
    adopted, rounds, _ = size_aware_edge_coloring(
        pairs, seed, weights, max(seed.values()) + 1)
    searched = anneal_coloring(pairs, weights, adopted, rounds + 1, 2_000_000)
    assert searched != adopted, (
        "the search returns its input on this mesh, so the gate below would "
        "compare one schedule with itself")


def test_a_different_schedule_gives_the_same_answer(monkeypatch):
    """The schedule decides which rows travel in which round. Regrouping the
    rounds moves no data and must move no bits; if it does, the new schedule
    is delivering the wrong rows and everything downstream is wrong quietly.
    """
    _need(4)
    _assert_the_search_changes_this_mesh(4)
    on, st_on, dt = _build(4, anneal="2", monkeypatch=monkeypatch)
    out_on = on(st_on, dt)
    off, st_off, _ = _build(4, anneal="0", monkeypatch=monkeypatch)
    out_off = off(st_off, dt)
    for name, a, b in (("u", out_on.u.data, out_off.u.data),
                       ("T", out_on.T.data, out_off.T.data),
                       ("p_s", out_on.p_s.data, out_off.p_s.data)):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"the annealed schedule changed {name}")
