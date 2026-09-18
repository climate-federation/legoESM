"""`--gm-treguier` reaches the MPAS lane, and only the lanes that implement it.

The flag used to be refused on every grid but tripole, because the Voronoi
GM/Redi raised for the Treguier block. MPAS implements it now (the SHARED
variant, not the ``nemo_native`` one the ORCA1-faithful tripole card runs), so
the refusal has to move rather than simply go away: FESOM still has no GM/Redi
at all, and the OPERATOR flags are still structured-grid constructions.
"""
from __future__ import annotations

import inspect

import pytest

from scripts.run.run_omip_core2 import _build_arg_parser, build_mpas_ocean


def _args(extra):
    return _build_arg_parser().parse_args(
        ["--mesh", "/some/mesh.nc", "--output", "/some/out"] + extra)


def _main_guard_source():
    """The grid guards live in main(); reach them without running a model."""
    from scripts.run import run_omip_core2
    return inspect.getsource(run_omip_core2.main)


class TestGridGuard:

    def test_mpas_is_now_allowed(self):
        src = _main_guard_source()
        assert 'args.gm_treguier and args.grid not in ("tripole", "mpas")' in src, (
            "the --gm-treguier grid guard no longer names mpas; either it was "
            "narrowed back or it was widened to grids with no GM/Redi")

    def test_fesom_is_still_refused(self):
        src = _main_guard_source()
        # fesom must NOT appear in the allowed tuple — it has no gm_redi block,
        # so the flag would be silently discarded there.
        line = [ln for ln in src.splitlines()
                if "args.gm_treguier and args.grid not in" in ln]
        assert len(line) == 1
        assert "fesom" not in line[0]

    def test_operator_flags_stay_tripole_only(self):
        """The slope/bolus operator set is NOT part of this widening.

        ``nemo_iso_lap`` slopes and the native slope positions are structured
        constructions; MPAS gained the COEFFICIENT, not the operator.
        """
        src = _main_guard_source()
        assert '_gm_op_flags and args.grid != "tripole"' in src


class TestBuilderWiring:

    def test_builder_takes_the_flag(self):
        sig = inspect.signature(build_mpas_ocean)
        for name in ("gm_treguier", "gm_aei0", "gm_kappa_min"):
            assert name in sig.parameters, name
        assert sig.parameters["gm_treguier"].default is False

    def test_call_site_forwards_it(self):
        """A parameter nothing passes is a parameter that does nothing."""
        from scripts.run import run_omip_core2
        src = inspect.getsource(run_omip_core2.main)
        assert "gm_treguier=args.gm_treguier" in src
        assert "gm_aei0=args.gm_aei0" in src
        assert "gm_kappa_min=args.gm_kappa_min" in src

    def test_builder_changes_exactly_one_field(self):
        """The Treguier block must be spliced onto THIS lane's own base.

        The tripole builder carries a comment recording why: an earlier
        version used the lat-lon default as its base and changed four fields
        at once, so no arm could be attributed to the coefficient. This pins
        that the MPAS builder splices `treguier` onto `config.gm_redi` rather
        than constructing a fresh block.
        """
        src = inspect.getsource(build_mpas_ocean)
        assert "_base = config.gm_redi" in src
        assert "_base._replace(treguier=_treg)" in src

    def test_builder_refuses_rather_than_disabling_visbeck(self):
        src = inspect.getsource(build_mpas_ocean)
        # phrase chosen to survive the comment's line wrapping
        assert "Disable Visbeck explicitly" in src
        # and it must not silently flip the other scheme off
        assert "visbeck=VisbeckConfig(enabled=False)" not in src
        assert "_base.visbeck.enabled" in src


def test_card_selects_it():
    """A knob ships with a card that selects it, or it is not shipped."""
    import pathlib
    card = (pathlib.Path(__file__).resolve().parents[2]
            / "scripts/cluster/omip_nemo/_mpas_gmtreguier_d30.sbatch")
    assert card.is_file(), card
    text = card.read_text()
    assert "--gm-treguier" in text
    assert "--grid mpas" in text
    # and it must not write into the baseline card's output directory
    assert "mpas_orca1faithtint_d${DAYS}" not in text
