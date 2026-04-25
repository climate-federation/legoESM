"""Tests for ocean test-matrix CLI overrides used by Eady experiments.

Specifically guards the wiring added for issue #213:
``--barotropic-div-damp`` (issue #213 hi-res SOM Eady investigation
recipe) must be parsed by ``cli.build_parser()``, materialised into
``config.BAROTROPIC_DIV_DAMP_OVERRIDE``, and threaded into
``EadyUniformConfig`` by the eady_uniform runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The ``ocean_test_matrix`` package lives under ``scripts/`` (sibling of
# ``src/legoesm``); add it to sys.path so we can import its modules
# directly in the tests.
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def _restore_overrides():
    """Snapshot/restore module-level overrides so a failing test doesn't
    leak state into the next one."""
    from ocean_test_matrix import config
    snapshot = {
        name: getattr(config, name)
        for name in (
            "B_H_OVERRIDE", "C_SMAG_OVERRIDE", "K_H_OVERRIDE",
            "U_SURFACE_OVERRIDE", "BAROTROPIC_DIV_DAMP_OVERRIDE",
            "TRACER_ADVECTION_OVERRIDE", "DEFAULT_DT", "DEFAULT_NLEV",
        )
    }
    yield
    for name, val in snapshot.items():
        setattr(config, name, val)


class TestBarotropicDivDampOverride:
    """Issue #213: ``--barotropic-div-damp`` CLI knob."""

    def test_override_default_is_none(self):
        from ocean_test_matrix import config
        assert config.BAROTROPIC_DIV_DAMP_OVERRIDE is None

    def test_cli_parses_flag(self):
        from ocean_test_matrix.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["--barotropic-div-damp", "0.123"])
        assert args.barotropic_div_damp == pytest.approx(0.123)

    def test_cli_help_lists_flag(self):
        from ocean_test_matrix.cli import build_parser
        help_text = build_parser().format_help()
        assert "--barotropic-div-damp" in help_text
        assert "issue #213" in help_text

    def test_runner_threads_override_into_eady_config(self):
        """The eady_uniform runner must propagate
        ``BAROTROPIC_DIV_DAMP_OVERRIDE`` into ``EadyUniformConfig`` —
        guarding the issue #213 investigation recipe end-to-end."""
        from ocean_test_matrix import config, experiments
        from legoesm.ocean.experiments.eady_uniform import EadyUniformConfig

        # Capture EadyUniformConfig at the point _create_ocean_setup is
        # called; abort the runner immediately afterwards so no compute
        # happens.
        captured: list[EadyUniformConfig] = []

        def _capture(tc, **kwargs):  # pragma: no cover - bypass real setup
            # The runner passes ``barotropic_div_damp=eu_config.barotropic_div_damp``
            # explicitly — but we want to assert on the eu_config that
            # ``overrides`` produced.  Stop the call here.
            captured.append(kwargs)
            raise StopIteration("captured")

        # Set both old and new overrides to non-default values to make
        # sure the new wiring doesn't shadow the existing ones.
        config.B_H_OVERRIDE = 5.0e10
        config.C_SMAG_OVERRIDE = 0.2
        config.BAROTROPIC_DIV_DAMP_OVERRIDE = 0.42
        config.TRACER_ADVECTION_OVERRIDE = "som"

        # Build a minimal latlon_channel TestCase.
        from ocean_test_matrix.testcase import TestCase
        tc = TestCase(
            case="eady_uniform", grid_type="latlon_channel",
            resolution="24x72", duration_days=1.0, quick_days=0.5,
        )

        # Monkey-patch _create_ocean_setup to capture and abort.
        original = experiments._create_ocean_setup
        experiments._create_ocean_setup = _capture
        try:
            with pytest.raises(StopIteration):
                experiments.run_eady_uniform(tc, Path("/tmp/_test_dummy"), days=1.0)
        finally:
            experiments._create_ocean_setup = original

        assert captured, "runner did not call _create_ocean_setup"
        kw = captured[0]
        # All four overrides must have made it through.
        assert kw["B_h"] == pytest.approx(5.0e10)
        assert kw["C_smag"] == pytest.approx(0.2)
        assert kw["barotropic_div_damp"] == pytest.approx(0.42)
        assert kw["tracer_advection"] == "som"
