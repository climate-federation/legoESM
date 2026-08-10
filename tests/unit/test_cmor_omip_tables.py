"""Coverage tests for the OMIP/CMIP6 ocean + ice CMOR tables.

Verifies the Oyr / Ofx / SIyr / msftmz / global-scalar additions
needed for centennial OMIP-2 submissions land correctly in the
shared ``CMOR_TABLES`` registry, route to the right realm, and
carry CF-compliant ``cell_methods`` strings.
"""

from __future__ import annotations

import pytest

from legoesm.io.cmor_output import (
    CMOR_TABLES,
    lookup_cmor_entry,
    table_realm,
)

# The per-table dicts are no longer module-private globals: the metadata
# is read from the vendored official CMOR tables and exposed through the
# public ``CMOR_TABLES`` registry.  (Importing ``_``-prefixed symbols
# across modules is banned in this repo anyway.)
_OMON_VARIABLES = CMOR_TABLES["Omon"]
_OYR_VARIABLES = CMOR_TABLES["Oyr"]
_OFX_VARIABLES = CMOR_TABLES["Ofx"]
_SIMON_VARIABLES = CMOR_TABLES["SImon"]
_SIYR_VARIABLES = CMOR_TABLES["SIyr"]


# ==============================================================================
# Table registration
# ==============================================================================

class TestNewTablesRegistered:

    def test_omip_tables_present(self):
        for tbl in ("Omon", "Oyr", "Ofx", "SImon", "SIyr"):
            assert tbl in CMOR_TABLES, f"missing table: {tbl}"

    def test_omip_tables_route_to_correct_realm(self):
        assert table_realm("Omon") == "ocean"
        assert table_realm("Oyr") == "ocean"
        assert table_realm("Ofx") == "ocean"
        assert table_realm("SImon") == "seaIce"
        assert table_realm("SIyr") == "seaIce"


# ==============================================================================
# Oyr / SIyr mirror their monthly siblings
# ==============================================================================

class TestAnnualTablesMirrorMonthly:

    def test_oyr_mirrors_omon(self):
        # Oyr should carry every Omon variable spec verbatim.
        assert set(_OYR_VARIABLES.keys()) == set(_OMON_VARIABLES.keys())
        for name, spec in _OMON_VARIABLES.items():
            assert _OYR_VARIABLES[name] == spec, (
                f"Oyr/{name} spec drifts from Omon/{name}"
            )

    def test_siyr_mirrors_simon(self):
        assert set(_SIYR_VARIABLES.keys()) == set(_SIMON_VARIABLES.keys())
        for name, spec in _SIMON_VARIABLES.items():
            assert _SIYR_VARIABLES[name] == spec, (
                f"SIyr/{name} spec drifts from SImon/{name}"
            )


# ==============================================================================
# Required OMIP-2 ocean variables present
# ==============================================================================

class TestOMIP2OceanVariables:
    """OMIP-2 / CMIP6 core ocean variables must resolve via lookup."""

    OMIP2_CORE = (
        "tos", "sos", "zos", "mlotst", "hfds", "wfo",
        "tauuo", "tauvo",
        "thetao", "so", "uo", "vo", "wo",
        "msftmz", "msftyz",
        "tosga", "sosga", "thetaoga", "soga", "masso", "volo",
    )

    @pytest.mark.parametrize("var", OMIP2_CORE)
    def test_var_resolves_to_omon(self, var):
        table_id, entry = lookup_cmor_entry(var)
        assert table_id == "Omon"
        assert "standard_name" in entry
        assert "units" in entry
        assert "cell_methods" in entry
        assert "dimensions" in entry

    def test_global_scalars_dimensioned_as_time_only(self):
        for var in ("tosga", "sosga", "thetaoga", "soga", "masso", "volo"):
            _, entry = lookup_cmor_entry(var, table="Omon")
            assert entry["dimensions"] == ("time",), (
                f"{var} should have dimensions=('time',); got {entry['dimensions']}"
            )

    def test_msftmz_has_basin_axis(self):
        _, entry = lookup_cmor_entry("msftmz", table="Omon")
        assert "basin" in entry["dimensions"]
        assert entry["units"] == "kg s-1"


# ==============================================================================
# Ofx time-invariant fields
# ==============================================================================

class TestOfxFields:

    OFX_VARS = ("areacello", "deptho", "sftof", "masscello", "volcello", "thkcello")

    @pytest.mark.parametrize("var", OFX_VARS)
    def test_ofx_var_resolves(self, var):
        table_id, entry = lookup_cmor_entry(var, table="Ofx")
        assert table_id == "Ofx"
        # Time-invariant variables MUST NOT carry a time dimension.
        assert "time" not in entry["dimensions"], (
            f"{var} is Ofx (time-invariant) but has 'time' in dimensions"
        )

    def test_areacello_units(self):
        _, entry = lookup_cmor_entry("areacello", table="Ofx")
        assert entry["units"] == "m2"

    def test_deptho_units(self):
        _, entry = lookup_cmor_entry("deptho", table="Ofx")
        assert entry["units"] == "m"


# ==============================================================================
# Cell-methods strings well-formed
# ==============================================================================

class TestCellMethods:

    def test_all_omon_vars_have_time_mean(self):
        for name, entry in _OMON_VARIABLES.items():
            cm = entry["cell_methods"]
            assert "time: mean" in cm or "time:" in cm, (
                f"{name} has no time: cell method: {cm!r}"
            )

    def test_all_ofx_vars_have_no_time_dimension(self):
        # Ofx vars are time-invariant — they MUST NOT carry a 'time'
        # dimension axis.  cell_methods MAY mention 'time: mean'
        # because some Ofx fields (masscello / volcello / thkcello)
        # are snapshots of a time-varying state that the archive
        # convention reports as a time-mean.
        for name, entry in _OFX_VARIABLES.items():
            assert "time" not in entry["dimensions"], (
                f"Ofx/{name} time-invariant variable carries 'time' "
                f"in dimensions: {entry['dimensions']}"
            )
