"""Tests for the shared recipe×setup selector mechanics (#388)."""
from __future__ import annotations

import pytest

from legoesm.core.setup_selector import (
    SETUP_KEYS,
    MatrixRunnerSpec,
    build_matrix_command,
    require_positive_finite,
    setup_signature,
    validate_setup,
)

# ocean-like spec (full CLI)
OCEAN = MatrixRunnerSpec(runner_path="scripts/matrix/run_ocean_test_matrix.py",
                         valid_grids=("latlon", "cubed_sphere", "latlon_channel"))
# atmosphere-like spec (case flag is --test, no --levels / --dt)
ATM = MatrixRunnerSpec(runner_path="scripts/matrix/run_atmosphere_test_matrix.py",
                       valid_grids=("cubed_sphere", "latlon", "spectral"),
                       case_flag="--test", levels_flag=None, dt_flag=None)
# sea-ice-like spec: case flag is --test and ALREADY exact (no `=` prefix)
ICE = MatrixRunnerSpec(runner_path="scripts/matrix/run_sea_ice_test_matrix.py",
                       valid_grids=("cubed_sphere", "column"),
                       case_flag="--test", exact_prefix="",
                       levels_flag=None, dt_flag=None, days_flag=None,
                       resolution_flag=None)


class TestValidate:
    def test_valid(self):
        validate_setup({"name": "lock_exchange", "grid": "latlon"}, OCEAN)

    def test_non_mapping(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            validate_setup("x", OCEAN)

    def test_unknown_field(self):
        with pytest.raises(ValueError, match="unknown setup field"):
            validate_setup({"name": "c", "grid": "latlon", "gird": 1}, OCEAN)

    def test_bad_name(self):
        with pytest.raises(ValueError, match="setup.name must be"):
            validate_setup({"grid": "latlon"}, OCEAN)

    def test_grid_not_in_matrix_set(self):
        with pytest.raises(ValueError, match="not a valid matrix grid"):
            validate_setup({"name": "c", "grid": "nope"}, OCEAN)

    def test_known_names(self):
        validate_setup({"name": "c", "grid": "latlon"}, OCEAN, known_names={"c"})
        with pytest.raises(ValueError, match="not a known experiment"):
            validate_setup({"name": "z", "grid": "latlon"}, OCEAN, known_names={"c"})

    def test_unsupported_override_rejected(self):
        # ATM has no --levels / --dt; supplying them must fail at the boundary
        with pytest.raises(ValueError, match="not supported by"):
            validate_setup({"name": "c", "grid": "latlon", "levels": 20}, ATM)
        with pytest.raises(ValueError, match="not supported by"):
            validate_setup({"name": "c", "grid": "latlon", "dt_seconds": 30.0}, ATM)
        # but the ocean spec accepts them
        validate_setup({"name": "c", "grid": "latlon", "levels": 20}, OCEAN)

    def test_bad_run_controls(self):
        for bad in ({"levels": 0}, {"levels": True}, {"dt_seconds": -1},
                    {"dt_seconds": float("nan")}, {"duration_days": 0},
                    {"quick": "yes"}):
            with pytest.raises(ValueError):
                validate_setup({"name": "c", "grid": "latlon", **bad}, OCEAN)

    def test_require_positive_finite(self):
        require_positive_finite("x", None)
        require_positive_finite("x", 3.0)
        for bad in (0, -1, True, float("inf"), float("nan"), "1"):
            with pytest.raises(ValueError):
                require_positive_finite("x", bad)

    def test_keys_constant(self):
        assert "name" in SETUP_KEYS and "grid" in SETUP_KEYS


class TestCommand:
    def test_ocean_exact_match(self):
        cmd = build_matrix_command(OCEAN, {"name": "lock_exchange", "grid": "latlon"},
                                   output_path="out/")
        assert "run_ocean_test_matrix.py" in cmd
        assert "--only =lock_exchange" in cmd
        assert "--grid latlon" in cmd and "--output out/" in cmd
        for flag in ("--levels", "--dt ", "--days", "--quick", "--resolution"):
            assert flag not in cmd, flag

    def test_atmosphere_uses_test_flag(self):
        cmd = build_matrix_command(ATM, {"name": "williamson2", "grid": "cubed_sphere"})
        assert "run_atmosphere_test_matrix.py" in cmd
        assert "--test =williamson2" in cmd          # case flag is --test
        assert "--only" not in cmd                    # not the equation-set flag
        assert "--grid cubed_sphere" in cmd

    def test_sea_ice_exact_no_prefix(self):
        # sea-ice --test is already exact: no leading `=`.
        cmd = build_matrix_command(ICE, {"name": "stefan_growth", "grid": "column"})
        assert "run_sea_ice_test_matrix.py" in cmd
        assert "--test stefan_growth" in cmd
        assert "=stefan_growth" not in cmd        # NO `=` prefix
        assert "--grid column" in cmd

    def test_overrides_mapped(self):
        cmd = build_matrix_command(OCEAN, {
            "name": "c", "grid": "latlon", "levels": 20, "dt_seconds": 30.0,
            "duration_days": 0.5, "resolution": "72x144", "quick": True})
        assert "--levels 20" in cmd and "--dt 30.0" in cmd
        assert "--days 0.5" in cmd and "--resolution 72x144" in cmd and "--quick" in cmd

    def test_signature_command_effective(self):
        a = setup_signature(OCEAN, {"name": "c", "grid": "latlon"})
        b = setup_signature(OCEAN, {"name": "c", "grid": "cubed_sphere"})
        assert a != b
        c = setup_signature(OCEAN, {"name": "c", "grid": "latlon", "quick": False})
        assert c == a  # disabled control is a no-op
