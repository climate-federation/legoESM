"""CMIP6 publication-compliance gate for the CMOR writer.

One test walks every variable in a written file and checks the FULL
attribute set against the vendored official CMOR table, so the whole
class of "hand-typed metadata drifted from the tables" defects fails
here instead of at ESGF.

Covers, per written file:
  * ``units``, ``standard_name``, ``long_name`` == the table entry
  * ``cell_methods`` == the table entry (incl. the ``area:`` clause the
    writer used to omit on 29 of 35 ``Amon`` variables)
  * ``positive`` present with the table's value on every flux variable
  * ``cell_measures`` == the table entry
  * on-disk dtype matches the table's ``type`` (CMIP6 ``real`` -> float32)
  * ``_FillValue`` and ``missing_value`` == 1.0e20 on data variables,
    and ABSENT on coordinate variables
  * every CV ``required_global_attributes`` entry present and non-empty
  * ``Conventions`` matches the CV regex
  * the DRS filename carries a time range for time-dependent tables

Non-vacuity: each assertion below was demonstrated RED against the
pre-fix writer (see the commit that introduced this file).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from legoesm.io import cmor_table_loader as tables  # noqa: E402
from legoesm.io.cmor_output import CFWriter  # noqa: E402

# CMIP6 fill/missing sentinel (every official table Header: "1e20").
FILL = 1.0e20

# Variables the AMIP lane writes, by table, with the shape they need.
_AMON_2D = (
    "clivi", "clt", "clwvi", "evspsbl", "hfls", "hfss", "hurs", "huss", "pr",
    "prc", "prsn", "prw", "ps", "psl", "rlds", "rldscs", "rlus", "rlut",
    "rlutcs", "rsds", "rsdscs", "rsdt", "rsus", "rsuscs", "rsut", "rsutcs",
    "rtmt", "sfcWind", "tas", "tasmax", "tasmin", "tauu", "tauv", "ts",
    "uas", "vas",
)
_AMON_3D = ("cli", "clw", "hur", "hus", "ta", "ua", "va", "wap", "zg")


def _amon_flux_vars():
    """Amon variables the official table declares a ``positive`` for.

    DERIVED from the table rather than hand-listed: vendoring a new flux
    entry (``rtmt``, ``rsuscs``, ...) then automatically extends this gate
    instead of silently leaving the new variable unchecked.
    ``test_flux_var_set_is_covered`` asserts the fixture writes all of them,
    so a table entry can never join the set without being exercised.
    """
    entries = tables.load_table("Amon")
    return tuple(
        sorted(v for v, e in entries.items() if e.get("positive"))
    )


FLUX_VARS = _amon_flux_vars()


@pytest.fixture(scope="module")
def amip_files(tmp_path_factory):
    """Write one file per AMIP Amon/day/CFday/fx variable; return paths."""
    out = tmp_path_factory.mktemp("cmor")
    lat = np.linspace(-87.5, 87.5, 8)
    lon = np.linspace(0.0, 337.5, 16)
    plev = tables.coordinate_axis("plev19")["requested"]
    plev = np.array([float(p) for p in plev])

    writer = CFWriter(
        output_dir=str(out),
        experiment_id="amip",
        model_id="legoESM-1-0",
        freq="mon",
        calendar="noleap",
        ref_date="1979-01-01",
    )
    paths = {}
    for i, var in enumerate(_AMON_2D):
        data = np.full((8, 16), 1.0 + i, dtype=np.float64)
        paths[("Amon", var)] = writer.write_field(
            var, data, time=15.0, time_bounds=(0.0, 31.0),
            lat=lat, lon=lon, table="Amon",
        )
    for var in _AMON_3D:
        data = np.full((plev.size, 8, 16), 2.0, dtype=np.float64)
        paths[("Amon", var)] = writer.write_field(
            var, data, time=15.0, time_bounds=(0.0, 31.0),
            lat=lat, lon=lon, plev=plev, table="Amon",
        )
    for var in ("pr", "psl", "rlut", "tas", "tasmax", "tasmin"):
        paths[("day", var)] = writer.write_field(
            var, np.full((8, 16), 3.0), time=0.5, time_bounds=(0.0, 1.0),
            lat=lat, lon=lon, table="day",
        )
    for var in ("ua", "va"):
        paths[("day", var)] = writer.write_field(
            var, np.full((1, 8, 16), 4.0), time=0.5, time_bounds=(0.0, 1.0),
            lat=lat, lon=lon, plev=np.array([85000.0]), table="day",
        )
    paths[("CFday", "rsut")] = writer.write_field(
        "rsut", np.full((8, 16), 5.0), time=0.5, time_bounds=(0.0, 1.0),
        lat=lat, lon=lon, table="CFday",
    )
    for var in ("areacella", "orog", "sftlf"):
        paths[("fx", var)] = writer.write_fixed(
            var, np.full((8, 16), 6.0), lat=lat, lon=lon,
        )
    writer.close()
    return paths


def _open(path):
    return netCDF4.Dataset(str(path))


# ---------------------------------------------------------------------------
# The single full-attribute compliance walk (defects 1-6, 9, 10)
# ---------------------------------------------------------------------------

def collect_attribute_problems(paths):
    """Return every variable-attribute deviation from the official table.

    Shared by the compliance gate and by its synthetic-violation
    self-test, so the self-test exercises the SAME code the gate runs.
    """
    problems = []
    for (table_id, var), path in sorted(paths.items()):
        entry = tables.load_table(table_id)[var]
        with _open(path) as ds:
            if var not in ds.variables:
                problems.append(f"{table_id}/{var}: variable absent from file")
                continue
            v = ds.variables[var]
            attrs = {a: v.getncattr(a) for a in v.ncattrs()}

            for key in ("units", "standard_name", "long_name", "cell_methods"):
                want = entry[key]
                got = attrs.get(key)
                if got != want:
                    problems.append(
                        f"{table_id}/{var}.{key}: got {got!r}, table says {want!r}"
                    )

            want_pos = entry.get("positive", "")
            got_pos = attrs.get("positive")
            if want_pos:
                if got_pos != want_pos:
                    problems.append(
                        f"{table_id}/{var}.positive: got {got_pos!r}, "
                        f"table says {want_pos!r}"
                    )
            elif got_pos is not None:
                problems.append(
                    f"{table_id}/{var}.positive: present ({got_pos!r}) but the "
                    f"table declares none"
                )

            want_cm = entry.get("cell_measures", "")
            got_cm = attrs.get("cell_measures")
            if want_cm and got_cm != want_cm:
                problems.append(
                    f"{table_id}/{var}.cell_measures: got {got_cm!r}, "
                    f"table says {want_cm!r}"
                )
    return problems


def test_variable_attributes_match_official_table(amip_files):
    """Every data variable carries exactly the official table metadata."""
    problems = collect_attribute_problems(amip_files)
    assert not problems, "CMOR attribute mismatches:\n  " + "\n  ".join(problems)


def test_compliance_walk_is_non_vacuous(amip_files, tmp_path):
    """Synthetic-violation self-test: re-introduce the PRE-FIX metadata.

    Copies a compliant ``rsut`` file, rewrites its attributes back to
    exactly what the writer emitted before this change (``cell_methods``
    "time: mean", no ``positive``, no ``cell_measures``, ``cli`` units
    "1"), and asserts the walk flags every one.  Without this, a walk
    that silently checked nothing would pass just as happily.
    """
    import shutil

    src = amip_files[("Amon", "rsut")]
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    with netCDF4.Dataset(str(dst), "a") as ds:
        v = ds.variables["rsut"]
        v.setncattr("cell_methods", "time: mean")  # pre-fix value
        v.delncattr("positive")
        v.delncattr("cell_measures")
        v.setncattr("units", "1")  # the pre-fix cli/clw defect

    problems = collect_attribute_problems({("Amon", "rsut"): dst})
    flagged = " ".join(problems)
    assert "cell_methods" in flagged, problems
    assert "positive" in flagged, problems
    assert "cell_measures" in flagged, problems
    assert "units" in flagged, problems
    # And the compliant original is still clean, so the walk is not
    # simply flagging everything.
    assert collect_attribute_problems({("Amon", "rsut"): src}) == []


def test_flux_var_set_is_covered():
    """Every Amon entry declaring ``positive`` is actually written above.

    Keeps the derived ``FLUX_VARS`` honest: vendoring a flux variable but
    forgetting to write it would otherwise shrink this gate silently.
    """
    written = set(_AMON_2D) | set(_AMON_3D)
    uncovered = sorted(set(FLUX_VARS) - written)
    assert not uncovered, (
        f"Amon table declares ``positive`` for {uncovered} but the fixture "
        f"never writes them, so their ``positive`` is unchecked."
    )


def test_flux_variables_have_positive(amip_files):
    """Defect 1: every flux variable carries the table's ``positive``."""
    assert len(FLUX_VARS) >= 15, FLUX_VARS
    missing = []
    for var in FLUX_VARS:
        path = amip_files[("Amon", var)]
        want = tables.load_table("Amon")[var]["positive"]
        assert want in ("up", "down"), f"table has no positive for {var}"
        with _open(path) as ds:
            got = (
                ds.variables[var].getncattr("positive")
                if "positive" in ds.variables[var].ncattrs()
                else None
            )
        if got != want:
            missing.append(f"{var}: got {got!r}, want {want!r}")
    assert not missing, "missing/incorrect ``positive``:\n  " + "\n  ".join(missing)


def test_plev_variables_keep_bare_time_mean(amip_files):
    """The 6 plev Amon variables are CORRECT with bare ``time: mean``.

    Guards against an over-eager "add ``area:`` everywhere" fix.
    """
    for var in ("ta", "ua", "va", "hus", "hur", "zg"):
        with _open(amip_files[("Amon", var)]) as ds:
            assert ds.variables[var].getncattr("cell_methods") == "time: mean", var


def test_output_dtype_is_float(amip_files):
    """Defect 9: CMIP6 declares ``type: real`` -> float32 on disk."""
    wrong = []
    for (table_id, var), path in sorted(amip_files.items()):
        want_type = tables.load_table(table_id)[var].get("type", "real")
        with _open(path) as ds:
            got = ds.variables[var].dtype
        expect = {"real": np.float32, "double": np.float64, "integer": np.int32}[
            want_type
        ]
        if got != expect:
            wrong.append(f"{table_id}/{var}: {got} (table type={want_type})")
    assert not wrong, "wrong on-disk dtype:\n  " + "\n  ".join(wrong)


def test_fill_and_missing_value(amip_files):
    """Defect 10: ``_FillValue``/``missing_value`` are 1.0e20, not NaN."""
    problems = []
    for (table_id, var), path in sorted(amip_files.items()):
        with _open(path) as ds:
            v = ds.variables[var]
            for attr in ("_FillValue", "missing_value"):
                if attr not in v.ncattrs():
                    problems.append(f"{table_id}/{var}: no {attr}")
                    continue
                val = float(v.getncattr(attr))
                if not np.isclose(val, FILL, rtol=1e-6):
                    problems.append(f"{table_id}/{var}.{attr} = {val!r}, want {FILL}")
            # CF: coordinate variables must NOT carry _FillValue.
            for cname in ("time", "lat", "lon", "plev", "height"):
                if cname in ds.variables and "_FillValue" in ds.variables[
                    cname
                ].ncattrs():
                    problems.append(f"{table_id}/{var}: coord {cname} has _FillValue")
    assert not problems, "fill-value problems:\n  " + "\n  ".join(problems)


def test_required_global_attributes(amip_files):
    """Defects 5 + 6: CV required globals present, Conventions valid."""
    required = tables.required_global_attributes()
    conv_re = re.compile(tables.cv_conventions_regex())
    problems = []
    for (table_id, var), path in sorted(amip_files.items()):
        with _open(path) as ds:
            present = set(ds.ncattrs())
            for attr in required:
                if attr not in present:
                    problems.append(f"{table_id}/{var}: missing global {attr!r}")
                    continue
                val = ds.getncattr(attr)
                if isinstance(val, str) and not val.strip():
                    problems.append(f"{table_id}/{var}: global {attr!r} is empty")
            if "Conventions" in present and not conv_re.match(
                str(ds.getncattr("Conventions"))
            ):
                problems.append(
                    f"{table_id}/{var}: Conventions "
                    f"{ds.getncattr('Conventions')!r} fails the CV regex"
                )
    assert not problems, "global-attribute problems:\n  " + "\n  ".join(problems)


def test_external_variables_matches_cell_measures(amip_files):
    """Defect 4: ``external_variables`` and ``cell_measures`` agree."""
    for (table_id, var), path in sorted(amip_files.items()):
        with _open(path) as ds:
            ext = (
                ds.getncattr("external_variables")
                if "external_variables" in ds.ncattrs()
                else ""
            )
            cm = (
                ds.variables[var].getncattr("cell_measures")
                if "cell_measures" in ds.variables[var].ncattrs()
                else ""
            )
            if not ext:
                continue
            measures = {tok for tok in cm.replace(":", " ").split() if tok != "area"}
            missing = measures - set(ext.split())
            assert not missing, (
                f"{table_id}/{var}: cell_measures names {sorted(missing)} but "
                f"external_variables is {ext!r}"
            )


def test_filenames_carry_time_range(amip_files):
    """Defect 7: DRS filenames end with the time range for time tables."""
    bad = []
    for (table_id, var), path in sorted(amip_files.items()):
        stem = path.stem
        if table_id == "fx":
            # fx files legitimately have no time range.
            if re.search(r"_\d{6,8}-\d{6,8}$", stem):
                bad.append(f"{stem}: fx must NOT carry a time range")
            continue
        if not re.search(r"_\d{6,8}-\d{6,8}$", stem):
            bad.append(f"{stem}: no _YYYYMM-YYYYMM / _YYYYMMDD-YYYYMMDD suffix")
    assert not bad, "filename problems:\n  " + "\n  ".join(bad)


# ---------------------------------------------------------------------------
# Table-level invariants (defect 11, and the deviations we refuse to hide)
# ---------------------------------------------------------------------------

def test_day_table_has_no_invented_variables():
    """Defect 11: ``ua850``/``va850``/``day.rsut`` are not CMIP6 day vars."""
    day = tables.load_table("day")
    for bogus in ("ua850", "va850", "rsut"):
        assert bogus not in day, f"{bogus!r} is not a CMIP6 ``day`` variable"
    assert "ua" in day and "va" in day
    assert "rsut" in tables.load_table("CFday")


def test_known_deviations_are_exactly_these():
    """The documented deviations may not grow silently."""
    assert tables.ALEVEL_EMITTED_ON_PLEV == ("cli", "clw")
    assert set(tables.UNITS_DEVIATIONS) == {("Omon", "tos"), ("Omon", "tosga")}
    assert set(tables.LEGACY_TABLE_ALIASES) == {"Aday", "Oyr", "SIyr"}
    non_cmip6 = {
        (t, v) for t, vs in tables.NON_CMIP6_VARIABLES.items() for v in vs
    }
    assert non_cmip6 == {("Lmon", "nee"), ("Omon", "rhopoto"), ("Omon", "sic")}
