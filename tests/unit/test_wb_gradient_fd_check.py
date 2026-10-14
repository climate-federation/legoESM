"""Verdict logic of the finite-difference gradient check.

The classifier IS the instrument: it decides whether a disagreement between
autodiff and a finite difference is evidence about the adjoint, or just the
forward's own noise and round-off. Every test below is a case an independent
review produced against the first version, which got each of them wrong.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "scripts/validate")

from wb_gradient_fd_check import build_args, classify, noise_floor


def _row(h, fd, floor=0.0, finite=True):
    return {"h": h, "fd": fd, "noise_floor": floor, "finite": finite}


def test_a_correct_adjoint_matches_over_a_plateau():
    rows = [_row(1e-5, 2.02), _row(1e-4, 2.00), _row(1e-3, 1.97)]
    verdict, ratio = classify(2.0, rows)
    assert verdict == "AD_MATCHES_FD"
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_a_flipped_sign_is_reported_as_such():
    rows = [_row(1e-5, -2.0), _row(1e-4, -2.0)]
    verdict, ratio = classify(2.0, rows)
    assert verdict == "SIGN_DISAGREES"
    assert ratio < 0


def test_no_plateau_is_not_a_verdict():
    """A curved loss gives a different slope at every step size. The first
    version picked whichever row sat closest to the autodiff value, which
    CERTIFIES a match whenever any step produces one -- circular."""
    wandering = [_row(1e-6, 0.4), _row(1e-5, 3.0), _row(1e-4, 40.0),
                 _row(1e-3, 900.0)]
    assert classify(2.0, wandering)[0] == "NO_PLATEAU"
    # and the row that happens to sit at the autodiff value does not rescue it
    assert classify(3.0, wandering)[0] == "NO_PLATEAU"


def test_rows_below_the_noise_floor_carry_no_information():
    rows = [_row(1e-6, 0.5, floor=5.0), _row(1e-5, 0.5, floor=5.0)]
    assert classify(2.0, rows)[0] == "UNRESOLVED"
    clean = [_row(1e-6, 0.5), _row(1e-5, 0.5)]
    assert classify(2.0, clean)[0] == "MAGNITUDE_DISAGREES"


def test_a_nonfinite_perturbation_is_not_evidence_about_the_adjoint():
    """An overflowing forward says something about the FORWARD. Counting it
    made an infinite finite difference into a magnitude disagreement."""
    rows = [_row(1e-4, float("inf"), finite=False),
            _row(1e-3, float("nan"), finite=False)]
    assert classify(2.0, rows)[0] == "UNRESOLVED"


def test_a_single_usable_row_cannot_establish_a_plateau():
    assert classify(2.0, [_row(1e-4, 2.0)])[0] == "NO_PLATEAU"


def test_the_floor_counts_round_off_not_just_repeatability():
    """A deterministic forward repeats exactly, so jitter is zero -- but the
    difference of two nearly-equal large losses still loses digits, and that
    error is invisible to a repeat evaluation."""
    assert noise_floor(1e-6, jitter=0.0, baseline=0.0) == 0.0
    big = noise_floor(1e-6, jitter=0.0, baseline=1e16, n_terms=4)
    assert big > 1.0, "round-off in a 1e16 loss must not read as a zero floor"
    # and the floor grows as the step shrinks, which is why small h is fragile
    assert noise_floor(1e-8, 0.0, 1e16, 4) > noise_floor(1e-4, 0.0, 1e16, 4)
    # jitter and round-off both count
    assert (noise_floor(1e-4, jitter=1e-3, baseline=1.0)
            > noise_floor(1e-4, jitter=0.0, baseline=1.0))


def test_cli_defaults():
    a = build_args(["--config", "c.yaml"])
    assert (a.n_scenes, a.n_params, a.checkpoint) == (4, 6, None)
    assert [float(x) for x in a.steps.split(",")][0] == 1e-6


def test_a_stable_plateau_at_twice_the_slope_is_a_kink_not_a_bug():
    """A central difference straddling a where-branch returns the average of
    the two one-sided slopes; when one side is flat that is half the true
    slope. Reading it as a broken adjoint sends the work to the wrong place."""
    rows = [_row(1e-5, 1.0), _row(1e-4, 1.0)]
    verdict, ratio = classify(2.0, rows)
    assert verdict == "KINK_SUSPECTED"
    assert ratio == pytest.approx(2.0)


def test_a_row_just_above_the_floor_is_not_usable():
    """Within a small multiple of the resolution the difference IS the
    resolution. Over five decades some row clears a bare floor by luck."""
    marginal = [_row(1e-5, 2.0, floor=0.5), _row(1e-4, 2.0, floor=0.5)]
    assert classify(2.0, marginal)[0] == "UNRESOLVED"
    clear = [_row(1e-5, 2.0, floor=0.01), _row(1e-4, 2.0, floor=0.01)]
    assert classify(2.0, clear)[0] == "AD_MATCHES_FD"


def test_column_devices_flag_parses():
    a = build_args(["--config", "x.yaml", "--column-devices", "16"])
    assert a.column_devices == 16
    assert build_args(["--config", "x.yaml"]).column_devices == 0
