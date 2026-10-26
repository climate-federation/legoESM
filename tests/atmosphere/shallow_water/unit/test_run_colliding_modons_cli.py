"""CLI-default guard for the colliding-modons validator driver (#800).

The dedicated driver ``scripts/validate/run_colliding_modons.py`` must default
to the VALIDATED matrix modon config — now the shared ``MODON_*`` constants in
``shallow_water_fv3_cdgrid.py`` (single source of truth with the matrix
``test_num == 8``).  #800: the previous defaults (no hyperdiff, ``damp_v=0.030``,
non-duogrid grid) let cube-seam noise erupt and blew a plain run up at ~day 40 —
guarded here as a regression gate on the defaults + on ``main``'s wiring, not
the full 100-day integration (that lives in the matrix runner's
``colliding_modons`` case).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import scripts.validate.run_colliding_modons as _driver
from scripts.validate.run_colliding_modons import build_arg_parser

_DRIVER_SRC = Path(_driver.__file__).read_text()


def _modon_constants():
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        MODON_DAMP_V,
        MODON_DIV_DAMP_FACTOR,
        MODON_HYPERDIFF_FACTOR,
        MODON_HYPERDIFF_SCALING,
    )
    return (MODON_DIV_DAMP_FACTOR, MODON_DAMP_V,
            MODON_HYPERDIFF_FACTOR, MODON_HYPERDIFF_SCALING)


def test_defaults_match_shared_modon_constants():
    args = build_arg_parser().parse_args([])
    div_damp, damp_v, hd_factor, hd_scaling = _modon_constants()
    # Parity against the shared source of truth (NOT re-hardcoded literals),
    # so a matrix recalibration cannot silently re-desync the driver (#800).
    assert args.div_damp == div_damp
    assert args.damp_v == damp_v
    assert args.hyperdiff_factor == hd_factor
    assert args.hyperdiff_scaling == hd_scaling
    assert args.use_duogrid is True
    # And the canonical values are the validated stable ones.  hd_scaling == 2
    # (the #753 item-1 default flip: (ref/n)^2 keeps C96 stable at the seams;
    # ^4 erupts).  div_damp == 0.03 (2026-08-25 retune, dual-reviewed: 8.0
    # gave the divergent mode a ~5.5-minute e-fold at C36 and flattened the
    # day-1 dipole ~50x vs the other grids; 0.03 = ~one-day six-cell e-fold,
    # day-1 recovery + 100-day C36/C48 completion re-validated, jobs
    # 9499349/9499350/9499364).
    assert (hd_factor, hd_scaling, div_damp, damp_v) == (1.0, 2, 0.03, 0.010)


def test_default_scaling_law_coefficient_ratios_c36_c48_c96():
    """#753 item 1: the default (ref/n)^2 law's *coefficients* must deliver the
    extra face-seam damping at C96 (the eruption fix) while leaving the C48
    calibration exactly invariant — the property that makes the default flip
    safe.  This pins the coefficient ratios (a silent revert to ^4 fails here);
    the *physical* C96 seam stability rests on the 100-day matrix integration at
    C96 (the default matrix cube resolution is C36, which is stable under both
    laws and so does NOT exercise the fix)."""
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        cdgrid_hyperdiff_cube,
    )
    _, _, _, hd_scaling = _modon_constants()
    assert hd_scaling == 2
    # C48 (== ref_n): exponent-invariant, so the default is byte-identical to
    # the pre-#753 ^4 law there — no regression to the calibrated coarse case.
    assert cdgrid_hyperdiff_cube(48, scaling_exponent=hd_scaling) == (
        cdgrid_hyperdiff_cube(48, scaling_exponent=4)
    )
    # C96: the default delivers 4x the ^4 backstop (= (96/48)^2), the
    # empirically-needed extra damping that stops the seam eruption.
    assert cdgrid_hyperdiff_cube(96, scaling_exponent=hd_scaling) == pytest.approx(
        4.0 * cdgrid_hyperdiff_cube(96, scaling_exponent=4)
    )
    # C36 (< ref_n): the default gives LESS damping than ^4 (0.56x), so the
    # flip cannot over-damp the coarse cores — it preserves them better.
    assert cdgrid_hyperdiff_cube(36, scaling_exponent=hd_scaling) < (
        cdgrid_hyperdiff_cube(36, scaling_exponent=4)
    )


def test_instability_probe_overrides_still_available():
    # A user must still be able to deliberately reproduce the un-backstopped
    # eruption (the #800 failure mode) for diagnosis.
    args = build_arg_parser().parse_args(["--no-duogrid", "--hyperdiff-factor", "0"])
    assert args.use_duogrid is False
    assert args.hyperdiff_factor == 0.0


def test_hyperdiff_scaling_restricted_to_documented_laws():
    # Only the two documented resolution laws are selectable on the CLI (#753).
    for good in ("2", "4"):
        assert build_arg_parser().parse_args(["--hyperdiff-scaling", good])
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--hyperdiff-scaling", "3"])


def test_main_threads_the_stability_args_into_the_run():
    # The #800 bug lived in main()'s wiring (a non-duogrid grid + zero
    # hyperdiff), not only the defaults.  Source-guard that main() still threads
    # the parsed knobs through — a future edit hardcoding e.g. use_duogrid=False
    # would pass the defaults test but re-break #800.
    assert "use_duogrid=args.use_duogrid" in _DRIVER_SRC
    assert "omega=0.0" in _DRIVER_SRC
    assert "args.hyperdiff_factor * cdgrid_hyperdiff_cube(" in _DRIVER_SRC
    assert "damp_v=args.damp_v" in _DRIVER_SRC
    # No silent no-backstop fallback resurrecting the crash config.
    assert "hyperdiff = 0.0" not in _DRIVER_SRC
