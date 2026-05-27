"""Unit tests for ``select_n_outer_split`` (iter-233).

FV3-style trace-time outer-subcycle selector. Returns a Python int
chosen from a conservative max-wind estimate at trace time — the
scan-friendly alternative to mid-run dt-shrinkage (iter-228 stub).

Tests cover:
1. iter-183 production config → expected n_split.
2. Steady-plateau opt-down (max_wind_safe small) → 1.
3. LES regime (small dt, small dx) → 1.
4. F11 worst-case ceiling (max_wind_safe=300) → expected.
5. Return type is ``int`` (NOT ``jnp.array``).
6. Input validation: dt_outer/dx/max_wind_safe > 0, cfl_safe in (0, 1].
"""
from __future__ import annotations

import pytest

from legoesm.timestepping.split_explicit import select_n_outer_split


def test_iter183_production_returns_8():
    """iter-183 production (dt=20, dx=2000, default F11 ceiling +
    cfl 0.4): n_split = ceil(20·300/(0.4·2000)) = 8."""
    n = select_n_outer_split(20.0, 2000.0)
    assert n == 8, f"expected 8, got {n}"


def test_steady_plateau_optdown_returns_1():
    """When max_wind_safe is small (e.g. user knows iter-183 plateau
    max|w| < 0.05 m/s), n_split should drop to 1 — no wasted
    subcycling."""
    n = select_n_outer_split(20.0, 2000.0, max_wind_safe=1.0)
    assert n == 1, f"expected 1, got {n}"


def test_les_regime_returns_1():
    """LES regime (dx=500, dt=0.5): the small dt and matching dx
    keep n_split=1 even at the F11 worst-case wind."""
    n = select_n_outer_split(0.5, 500.0)
    assert n == 1, f"expected 1, got {n}"


def test_aggressive_cfl_target_inflates_n_split():
    """Tighter cfl_safe (0.2 vs default 0.4) ~doubles the required
    n_split for the same wind+grid+dt (exact factor 2 may be ±1 due
    to ceil() rounding)."""
    n_default = select_n_outer_split(20.0, 2000.0)  # ceil(7.5)=8
    n_tight = select_n_outer_split(20.0, 2000.0, cfl_safe=0.2)  # ceil(15.0)=15
    # n_tight should be at least 1.8× n_default (allows ceil
    # rounding slack: 2·7.5=15.0 vs 2·8=16 → tolerated 15 ≥ 14.4).
    assert n_tight >= 1.8 * n_default, (
        f"halving cfl_safe should ~double n_split; got "
        f"default={n_default}, tight={n_tight}."
    )


def test_iter183_wallce_ceiling_matches_docstring_example():
    """Direct table cross-check of the docstring examples."""
    # iter-183: 8
    assert select_n_outer_split(20.0, 2000.0) == 8
    # Steady opt-down: 1
    assert select_n_outer_split(20.0, 2000.0, max_wind_safe=1.0) == 1
    # LES: 1
    assert select_n_outer_split(0.5, 500.0) == 1


def test_return_type_is_python_int():
    """Output MUST be a Python ``int`` (NOT ``np.int64``,
    ``jnp.array``, or anything traced). FV3-style trace-time
    selection requires a compile-time constant."""
    n = select_n_outer_split(20.0, 2000.0)
    assert type(n) is int, f"expected type(n) is int, got {type(n)}"


def test_minimum_one_substep_even_at_zero_wind_estimate():
    """A very-small max_wind_safe must still return n_split >= 1
    (zero subcycling is the floor — running zero inner steps would
    advance the state by zero)."""
    n = select_n_outer_split(20.0, 2000.0, max_wind_safe=1e-10)
    assert n == 1, f"expected n>=1 floor, got {n}"


@pytest.mark.parametrize("bad_dt", [0.0, -1.0, -20.0])
def test_rejects_nonpositive_dt(bad_dt):
    with pytest.raises(ValueError, match="dt_outer"):
        select_n_outer_split(bad_dt, 2000.0)


@pytest.mark.parametrize("bad_dx", [0.0, -100.0])
def test_rejects_nonpositive_dx(bad_dx):
    with pytest.raises(ValueError, match="dx"):
        select_n_outer_split(20.0, bad_dx)


@pytest.mark.parametrize("bad_w", [0.0, -1.0])
def test_rejects_nonpositive_max_wind_safe(bad_w):
    with pytest.raises(ValueError, match="max_wind_safe"):
        select_n_outer_split(20.0, 2000.0, max_wind_safe=bad_w)


@pytest.mark.parametrize("bad_cfl", [0.0, -0.1, 1.1, 2.0])
def test_rejects_invalid_cfl_safe(bad_cfl):
    with pytest.raises(ValueError, match="cfl_safe"):
        select_n_outer_split(20.0, 2000.0, cfl_safe=bad_cfl)


def test_scaling_dt_outer_proportional_to_n_split():
    """Doubling dt_outer doubles n_split (keeps inner dt the same)."""
    n_1 = select_n_outer_split(10.0, 2000.0)
    n_2 = select_n_outer_split(20.0, 2000.0)
    assert n_2 == 2 * n_1, (
        f"doubling dt_outer should double n_split; got "
        f"dt=10 -> {n_1}, dt=20 -> {n_2}"
    )
