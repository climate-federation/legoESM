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
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
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
    # And the canonical values are the validated stable ones.
    assert (hd_factor, hd_scaling, div_damp, damp_v) == (1.0, 4, 8.0, 0.010)


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
