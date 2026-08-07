#!/usr/bin/env python
"""Benchmark an AMIP CMOR output tree against observations via ClimateEval.

Standalone consumer script: invoked as a subprocess under the *separate*
ClimateEval environment (iris + ESMValTool; never a legoESM dependency),
never imported into the legoESM/JAX process. See
``legoesm.driver.climateeval_hook.maybe_run_climateeval`` for the caller.

Runs one or more ClimateEval suites against the same CMOR tree and renders
them into a SINGLE combined portable HTML report (plus one ``.ddb`` per
suite). With no ``--suite``, runs ALL bundled suites (every tier); a suite
whose reference / model data is missing is **skipped and reported**, not
fatal, so the report always contains whatever could be scored.

Three properties here are load-bearing and each fixes a defect that made
the wrapper silently discard or corrupt work while still exiting 0 — see
the individual docstrings for the evidence:

* ``write_suite_yaml`` dumps with ``sort_keys=False``. Preprocessor order
  IS the pipeline; alphabetising it zeroed out the Tier-1 conservation
  suite entirely.
* ``apply_reference_fallback`` keeps each variable's own designated
  reference when it is staged and uses ERA5 only as a per-variable
  fallback. A blanket ERA5 rewrite deleted every TOA-radiation and
  surface-flux metric, because the staged ERA5 has none of those fields.
* ``cmor_nc_paths`` / ``group_blocks_by_frequency`` load the CMOR tables
  matching the frequency each diagnostic asks for. ClimateEval matches
  model cubes on ``var_name`` alone, so loading only the monthly tables
  fed monthly means to daily and sub-daily diagnostics, which then
  produced plausible but wrong numbers rather than an error.

``climateeval check`` runs before each benchmark group so an input problem
(missing variable, wrong units, non-overlapping timerange) is reported
before the expensive reference load rather than after it.

Graceful degradation is per-VARIABLE inside ClimateEval itself
(``SimpleDiagnostic._handle_metric_error``), but only if constructed with
``fail_on_metric_error=False`` -- ClimateEval's own default is ``True``,
which raises ``MetricError`` on the first variable a diagnostic cannot
score (e.g. ``AnnualMeanTimeSeries`` needs a multi-year timerange to build
a same-length model/reference series; a short run legitimately can't
satisfy that for that one diagnostic+variable). Left at the ClimateEval
default, that single failure propagates out of ``Suite.get_database()``
and is caught by this script's per-SUITE ``try/except`` below, discarding
every OTHER diagnostic in the suite (``map``/``zonal_line``/
``zonal_profile``/etc.) along with it -- so one inapplicable variable in
one diagnostic silently zeroed out the whole suite's report, including the
diagnostics that would have scored fine. ``--fail-on-metric-error`` is
therefore False by default here (log + skip that one variable, keep the
rest of the suite), matching the existing ``--fail-on-missing-data``
convention; pass it to restore ClimateEval's fail-hard behavior.

Usage:
    <climateeval-env>/bin/python scripts/validate/run_amip_climateeval.py \\
        --cmor-dir /scratch/.../amip_run/cmor \\
        --model-id legoESM-1-0 \\
        --data-root-dir /work/bd1179/b309141/climateeval_input \\
        --output-dir /scratch/.../amip_run
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

DEFAULT_REPORT_NAME = "climateeval_report.html"

# CMOR MIP tables that hold each output frequency. ClimateEval selects a model
# cube by var_name ALONE (``climateeval/_utils.py::get_prepared_cube`` ->
# ``data.extract_cube(NameConstraint(var_name=...))``) with no frequency filter,
# so the CubeList handed to a suite must contain exactly ONE cube per variable
# -- hence tables are chosen to match the frequency the suite asks for rather
# than all being loaded at once (two ``pr`` cubes would make extract_cube raise).
_CMOR_TABLES_BY_FREQUENCY: dict[str, tuple[str, ...]] = {
    "mon": ("Amon", "Omon", "SImon", "Lmon", "Emon"),
    "day": ("day", "CFday", "Oday", "SIday"),
    "3hr": ("3hr", "E3hr", "CF3hr"),
    "6hr": ("6hrLev", "6hrPlev", "6hrPlevPt"),
    "1hr": ("1hr", "E1hr", "AERhr"),
}
# Fixed fields (areacella/sftlf/orog) are frequency-independent and are needed
# by area weighting and land-sea masking, so they load alongside every group.
_FIXED_CMOR_TABLES = ("fx", "Ofx")
_DEFAULT_FREQUENCY = "mon"


ERA5_MONTHLY = "climateeval.data.ERA5Monthly"
ERA5_SUBMONTHLY = "climateeval.data.ERA5Hourly"


def era5_fallback_for(frequency: str) -> str:
    """ERA5 data-source class to fall back to for a given frequency.

    Both ERA5 classes share one ``DataSourceInformation`` (id
    ``reanalysis_ERA5``) and ClimateEval resolves the on-disk directory as
    ``<data_root>/<source id>/<frequency>/<var_name>``, so the choice only
    matters for the (disabled) CDS download path — but keeping it honest
    means a ``day``/``1hr`` variable never claims a *monthly* reference.
    """
    return ERA5_MONTHLY if frequency == "mon" else ERA5_SUBMONTHLY


def iter_variable_settings(suite_def: list[Any]) -> Iterator[dict[str, Any]]:
    """Yield each per-variable settings dict in a suite definition, once.

    A ClimateEval suite YAML is a list of diagnostic blocks, each with a
    ``variables`` list of per-variable settings dicts. Several diagnostics
    commonly share the *same* list object via a YAML anchor/alias (e.g.
    ``map`` and ``zonal_line`` both reusing ``&2d_variables``), and
    ``<<: *anchor`` merges share the *values* of the merged keys. Mutating
    a shared dict once is visible from every diagnostic that references it,
    which is what we want -- so this yields each dict object exactly ONCE,
    keyed on identity, to keep transforms idempotent and their logs honest.
    """
    seen: set[int] = set()
    for diagnostic_block in suite_def:
        if not isinstance(diagnostic_block, dict):
            continue
        variables = diagnostic_block.get("variables")
        if not isinstance(variables, list):
            continue
        for var_settings in variables:
            if isinstance(var_settings, dict) and id(var_settings) not in seen:
                seen.add(id(var_settings))
                yield var_settings


def apply_reference_fallback(
    suite_def: list[Any],
    is_available: Callable[[str, str, str], bool],
    *,
    era5_only: bool = False,
    strip_other_data: bool = True,
) -> tuple[list[Any], list[str]]:
    """Resolve each variable's reference to something actually on disk.

    This REPLACES an earlier blanket ``era5_only_suite_def`` transform that
    rewrote *every* ``reference_data`` to ``ERA5Monthly``. That was a real
    data-loss bug: ERA5 as staged has no ``rsut``, ``rlut``, ``hfls``,
    ``hfss`` or ``clwvi``, while the unmodified suites already point
    ``rsut``/``rlut``/``rtnt``/``swcre``/``lwcre`` at CERES-EBAF and
    ``hfls``/``hfss`` at MERRA2 — both of which ARE staged. The rewrite
    therefore deleted every TOA-radiation and surface-flux metric the suite
    would have produced (8 variables x 4 diagnostics in
    ``Tier2_atmosphere_monthly`` alone).

    The legitimate need it was serving (see
    ``docs/user-guide/climateeval_evaluation.md``) was that a personal data
    root typically stages only a SUBSET of ClimateEval's references, so a
    suite pointing at an unstaged one scores nothing. That need is met here
    per-VARIABLE instead of by fiat: the suite's designated reference is
    kept whenever it is available, and ERA5 is used only as a fallback when
    it is not.

    Parameters
    ----------
    suite_def:
        Parsed suite YAML (mutated in place and also returned).
    is_available:
        ``(source_str, var_name, frequency) -> bool`` predicate reporting
        whether that data source has that variable on disk. Injected so
        this transform stays pure and unit-testable without the ClimateEval
        environment installed.
    era5_only:
        Restore the old blanket-ERA5 behaviour (for reproducing a prior
        ERA5-only scorecard). Still availability-checked.
    strip_other_data:
        Drop ``other_data`` (multi-model intercomparison) entries. The
        documented point of ``--evaluate`` is "how far off is legoESM from
        the observations", not a CMIP6 intercomparison, so this stays on by
        default; unlike the reference rewrite it discards no observational
        metric.

    Returns
    -------
    :
        ``(suite_def, notes)`` where ``notes`` is a human-readable log of
        every reference decision, so the run's stdout says exactly which
        reference produced which number.
    """
    notes: list[str] = []
    for var_settings in iter_variable_settings(suite_def):
        if strip_other_data:
            var_settings.pop("other_data", None)

        designated = var_settings.get("reference_data")
        if not isinstance(designated, str):
            continue

        var_id = str(var_settings.get("id", var_settings.get("var_name", "?")))
        var_name = str(var_settings.get("var_name", var_id))
        frequency = str(var_settings.get("frequency", "mon"))
        fallback = era5_fallback_for(frequency)

        preferred = fallback if era5_only else designated
        if is_available(preferred, var_name, frequency):
            var_settings["reference_data"] = preferred
            if preferred != designated:
                notes.append(f"{var_id}: {designated} -> {preferred} (era5-only mode)")
            continue

        if preferred != fallback and is_available(fallback, var_name, frequency):
            var_settings["reference_data"] = fallback
            notes.append(
                f"{var_id}: {designated} unavailable -> fallback {fallback}",
            )
            continue

        # Neither the designated reference nor ERA5 has it. Leave the
        # designated reference in place: ClimateEval logs it as missing data
        # and skips that ONE variable (fail_on_missing_data=False), which is
        # the honest outcome — better than silently scoring against a
        # reference that does not contain the quantity.
        notes.append(
            f"{var_id}: NO reference available ({designated}, {fallback}) "
            f"-> variable will be skipped",
        )
    return suite_def, notes


def write_suite_yaml(suite_def: list[Any], path: Path) -> None:
    """Write a (possibly transformed) suite definition back out as YAML.

    ``sort_keys=False`` is LOAD-BEARING, not cosmetic. ClimateEval runs a
    variable's ``additional_preprocessors`` in **mapping insertion order**
    (``climateeval/_utils.py::_run_additional_preprocessors`` iterates
    ``variable.additional_preprocessors.items()``), so the YAML key order
    *is* the preprocessing pipeline order. PyYAML's ``yaml.dump`` default
    is ``sort_keys=True``, which alphabetises them and silently rewrites
    the pipeline:

    Tier1_consistency_checks, as authored::

        regrid -> area_statistics(sum) -> annual_statistics
              -> regrid_time -> anomalies

    after an alphabetising round-trip::

        annual_statistics -> anomalies -> area_statistics(sum)
              -> regrid -> regrid_time

    ``area_statistics`` collapses latitude/longitude to a scalar, so the
    later ``regrid`` sees a cube with no horizontal grid and ESMValCore
    rejects the area-weighted scheme ("Regridding scheme 'area_weighted'
    not available for unstructured data"). Every variable in the Tier-1
    conservation suite failed that way and the suite produced ZERO data
    rows. The same reordering also puts ``anomalies`` before
    ``area_statistics`` (a different physical quantity: the mean of local
    relative anomalies, not the relative anomaly of the global sum) and
    swaps ``regrid``/``mask_landsea`` in Tier2_ocean_monthly.
    """
    import yaml  # noqa: PLC0415 - deferred; see main()'s import block

    with path.open("w") as f:
        yaml.dump(suite_def, f, sort_keys=False)


def complex_diagnostics(suite_def: list[Any]) -> list[str]:
    """Names of the suite's diagnostics that are ClimateEval *complex* ones.

    Complex diagnostics (``climateeval.diags.complex.*``) take
    ``data: dict[str, CubeList]`` keyed by EXPERIMENT — ``ECS`` wants
    ``{'4xco2': ..., 'picontrol': ...}`` — whereas this wrapper scores one
    CMOR tree and passes a single flat ``CubeList``. Detected from the
    suite's own ``diagnostic`` class paths rather than a hardcoded suite
    name, so a future complex suite is handled too.
    """
    return [
        str(block.get("name", "?"))
        for block in suite_def
        if isinstance(block, dict)
        and ".diags.complex." in str(block.get("diagnostic", ""))
    ]


def suite_skip_reason(suite: str, experiment_id: str, suite_def: list[Any]) -> str | None:
    """Explain why a suite cannot be scored here, or ``None`` to run it.

    Skipping up front with a reason beats letting the run fail deep inside
    ClimateEval: passing a flat ``CubeList`` where a complex diagnostic
    expects ``{'4xco2': ..., 'picontrol': ...}`` produced a confusing
    ``ValueError: Expected keys ('4xco2', 'picontrol') for data dictionary,
    got (<iris 'Cube' of ...>, ...)`` that dumped every cube in the tree
    into the log and told the reader nothing about the real cause.
    """
    names = complex_diagnostics(suite_def)
    if not names:
        return None
    is_amip = experiment_id.lower().startswith("amip")
    applicability = (
        "an AMIP run is a single prescribed-SST experiment, so it provides "
        "neither the abrupt-4xCO2 nor the piControl leg such a diagnostic "
        "needs"
        if is_amip
        else "this wrapper scores ONE CMOR tree and cannot supply the "
             "per-experiment data dictionary such a diagnostic needs"
    )
    return (
        f"complex diagnostic(s) {names} not applicable for experiment "
        f"'{experiment_id}': {applicability}"
    )


class CheckOutcome(NamedTuple):
    """Result of ``climateeval check`` for one frequency group.

    ``statuses`` maps each REQUIRED MODEL variable to its check status
    (``pass``/``warn``/``fail``). It is empty when the check could not be
    run at all, which callers must treat as "no information" rather than
    "nothing is usable".
    """

    passed: bool
    statuses: dict[str, str]


def is_group_unscoreable(variable_statuses: Mapping[str, str]) -> bool:
    """True when a frequency group cannot produce a single model metric.

    A metric compares MODEL data against a reference, so if ``climateeval
    check`` finds no usable model variable at all, running the group can
    only ever emit reference-only rows — at, in one measured case, a very
    high price: ``Tier2_ocean_monthly`` on an atmosphere-only AMIP tree
    scored 0/6 required variables yet still spent ~4 h of a 4 h 25 m
    evaluation loading ORAS5 and area-weight-regridding its irregular grid
    for three ``mlotst`` variables the model does not output.

    Deliberately driven by MODEL data ONLY. It never consults the
    reference-availability probe, so a reference source that probe cannot
    see (``ORAS5``/``RAPID`` read from outside ``--data-root-dir``; see
    ``make_availability_checker``) can NEVER cause a scoreable group to be
    skipped — the invisible source is exactly the expensive one, and the
    decision to skip is taken without asking it anything.

    Fails open in both directions that matter:

    * an EMPTY mapping (the check crashed, or the suite requires no model
      input at all) returns False — never skip on ignorance;
    * a single ``pass`` or ``warn`` variable returns False, so a group
      keeps running whenever any model metric is still conceivable.
    """
    if not variable_statuses:
        return False
    return all(status == "fail" for status in variable_statuses.values())


def check_model_input(
    suite_yml: Path,
    cmor_dir: Path,
    timerange: str,
    frequencies: Iterable[str],
) -> CheckOutcome:
    """Run ``climateeval check`` on the model input before benchmarking.

    ClimateEval's own ``_check.check_model_data`` validates presence,
    coordinates, units, timerange coverage and levels from METADATA only
    (no preprocessing, no reference data, no array realization), so it is
    cheap. Running it first surfaces problems that otherwise only show up
    after a long benchmark — the reference-window mismatches in
    ``Tier3_dynamics`` (Hovmoller pinned to 20000101/20021231 and QBO to
    19900101/20091231 against a 1979 run) burned the full reference load
    before reporting nothing scoreable.

    ``check_model_data`` takes a single path, so the exact file set this
    frequency group will load is presented to it as a directory of
    symlinks — checking the same files that will actually be benchmarked,
    not the whole tree (whose duplicate ``pr`` across ``Amon``/``day``
    would be misreported as a concatenation failure).
    """
    from climateeval._check import check_model_data  # noqa: PLC0415

    paths = cmor_nc_paths(cmor_dir, frequencies)
    if not paths:
        return CheckOutcome(passed=False, statuses={})
    try:
        with tempfile.TemporaryDirectory(prefix="legoesm_ceval_check_") as tmp:
            link_dir = Path(tmp)
            for index, path in enumerate(paths):
                link = link_dir / path.name
                if link.exists():
                    link = link_dir / f"{index}_{path.name}"
                link.symlink_to(path)
            report = check_model_data(
                link_dir,
                suites=[suite_yml],
                timerange=timerange or None,
            )
    except Exception as exc:
        # The check is ADVISORY. It must never be the reason a suite that
        # would have scored gets skipped, so it fails open: an empty
        # `statuses` reads as "no information" to is_group_unscoreable.
        print(f"    check: skipped ({type(exc).__name__}: {exc})")
        return CheckOutcome(passed=True, statuses={})

    problems = [v for v in report.variables if v.status != "pass"]
    print(
        f"    check: {len(report.variables) - len(problems)}/"
        f"{len(report.variables)} required variable(s) usable "
        f"({report.n_cubes} cubes)",
    )
    for variable in problems:
        detail = "; ".join(
            f"{c.name}: {c.message}" for c in variable.checks if c.status != "pass"
        )
        print(f"      [{variable.status}] {variable.var_name}: {detail}")
    return CheckOutcome(
        passed=report.passed,
        statuses={v.var_name: v.status for v in report.variables},
    )


def make_availability_checker(
    data_root_dir: Path,
) -> Callable[[str, str, str], bool]:
    """Build the on-disk ``is_available`` predicate for the reference fallback.

    Mirrors ClimateEval's OWN resolution order in
    ``climateeval.data._base.DataSource.get_cube``: a variable is available
    if ``<data_root>/<source id>/<frequency>/<var_name>/*.nc`` exists, or —
    for a derived variable (``lwcre``, ``swcre``, ``rtnt``, ...) — if EVERY
    variable it is derived from is present under the same source. Downloads
    are deliberately not considered: this script runs with
    ``download_missing_data`` off by default and a network fetch is not
    "availability".

    Data source *generators* (``CMIP6HistoricalR1I1P1F1`` and friends)
    expand to a wildcard set of models and expose no single ``id``; they are
    reported unavailable, which matches this data root (no CMIP6 staged) and
    the documented intent of not doing a multi-model intercomparison here.

    KNOWN BLIND SPOT — a "False" here means "not under ``data_root_dir`` in
    the standard layout", NOT "ClimateEval cannot load it". A few sources
    override ``get_cube`` and read from paths outside ``--data-root-dir``
    entirely: ``ORAS5`` reads the raw NEMO + grid archive
    (``CLIMATEEVAL_ORAS5_*_DIR``) and ``RAPID`` a single CMORized file
    (``CLIMATEEVAL_RAPID_FILE``), both defaulting into ``/work/bd0854/DATA``.
    Those report unavailable here, so this transform leaves their designated
    reference in place (the deliberate "no reference available" branch) and
    ClimateEval then loads them anyway. That is CORRECT for scoring, but it
    is not free: the three ``mlotst`` variables in ``Tier2_ocean_monthly``
    each cost ~80 min of area-weighted regridding off ORAS5's irregular
    grid. Restrict with ``--suite`` when scoring an atmosphere-only AMIP
    run, which has no ocean output for them to be compared against.
    (``OSI450`` and ``WOA`` also override ``get_cube`` but resolve through
    the standard layout first, so they are probed correctly.)

    Imports ClimateEval, so it is only callable inside the ClimateEval
    environment; the transform it feeds stays pure and testable.
    """
    from climateeval._variable import VARIABLES, Variable  # noqa: PLC0415

    source_id_cache: dict[str, str | None] = {}

    def source_id(source_str: str) -> str | None:
        if source_str not in source_id_cache:
            source_id_cache[source_str] = _resolve_source_id(source_str)
        return source_id_cache[source_str]

    def has_files(sid: str, var_name: str, frequency: str) -> bool:
        return any((data_root_dir / sid / frequency / var_name).glob("*.nc"))

    def is_available(source_str: str, var_name: str, frequency: str) -> bool:
        sid = source_id(source_str)
        if sid is None:
            return False
        if has_files(sid, var_name, frequency):
            return True
        if not VARIABLES.get(var_name, {}).get("derived", False):
            return False
        try:
            required = Variable(var_name, var_name, frequency).get_required_variables()
        except Exception:  # unsupported variable / derivation not resolvable
            return False
        # ClimateEval only derives when EVERY required input is present.
        return bool(required) and all(
            has_files(sid, req.var_name, req.frequency) for req in required
        )

    return is_available


def _resolve_source_id(source_str: str) -> str | None:
    """Get the on-disk directory id of a ``climateeval.data.X`` class string."""
    try:
        module_name, class_name = source_str.rsplit(".", 1)
        source = getattr(importlib.import_module(module_name), class_name)()
        return str(source.id)
    except Exception:
        return None


def suite_db_path(output_dir: Path, suite: str) -> Path:
    """Canonical per-suite DuckDB path under the run's output directory."""
    return output_dir / f"climateeval_{suite}.ddb"


def block_frequencies(diagnostic_block: Any) -> frozenset[str]:
    """Output frequencies a single diagnostic block's variables ask for."""
    if not isinstance(diagnostic_block, dict):
        return frozenset({_DEFAULT_FREQUENCY})
    variables = diagnostic_block.get("variables")
    if not isinstance(variables, list) or not variables:
        # Complex diagnostics (ECS) declare no variables; they read whole
        # experiments, not a frequency-selected CubeList.
        return frozenset({_DEFAULT_FREQUENCY})
    return frozenset(
        str(v.get("frequency", _DEFAULT_FREQUENCY))
        for v in variables
        if isinstance(v, dict)
    ) or frozenset({_DEFAULT_FREQUENCY})


def group_blocks_by_frequency(
    suite_def: list[Any],
) -> list[tuple[frozenset[str], list[Any]]]:
    """Split a suite into runs that each need ONE frequency-selected CubeList.

    ClimateEval matches model cubes on ``var_name`` only, so a suite mixing
    frequencies (``Tier3_dynamics``: daily Hovmoller + monthly QBO) cannot be
    served by a single flat CubeList — loading ``Amon`` and ``day`` together
    gives two ``pr`` cubes and ``extract_cube`` raises. Grouping at the
    DIAGNOSTIC-BLOCK level works because no bundled suite mixes frequencies
    *within* one block; the groups are then run in sequence into the same
    ``.ddb`` (diagnostic names are unique per suite, so their schemas never
    collide). Insertion order is preserved so the report reads as authored.
    """
    groups: dict[frozenset[str], list[Any]] = {}
    for diagnostic_block in suite_def:
        groups.setdefault(block_frequencies(diagnostic_block), []).append(
            diagnostic_block,
        )
    return list(groups.items())


def frequency_cmor_tables(frequencies: Iterable[str]) -> tuple[str, ...]:
    """CMOR table directories holding the given frequencies (no fixed fields)."""
    tables: list[str] = []
    for frequency in sorted(set(frequencies)):
        tables.extend(_CMOR_TABLES_BY_FREQUENCY.get(frequency, ()))
    return tuple(dict.fromkeys(tables))


def cmor_tables_for(frequencies: Iterable[str]) -> tuple[str, ...]:
    """CMOR table directories to load: the frequencies' plus the fixed fields."""
    return (*frequency_cmor_tables(frequencies), *_FIXED_CMOR_TABLES)


def cmor_nc_paths(
    cmor_dir: Path,
    frequencies: Iterable[str] = (_DEFAULT_FREQUENCY,),
) -> list[Path]:
    """Resolve the CMOR ``*.nc`` files to load for the given frequencies.

    If ``cmor_dir`` holds ``*.nc`` directly (e.g. a single ``…/cmor/Amon``
    table), load those (back-compat: the caller has already selected the
    table). Otherwise treat it as the ``cmor`` root and load only the MIP
    tables matching *frequencies*, plus the fixed fields.

    Loading ONLY the monthly tables regardless of what the suite asked for
    was a silent-wrong-answer bug: a sub-daily or daily diagnostic was
    served the MONTHLY cube (ClimateEval does not filter on frequency), so
    e.g. ``Tier3_dynamics``'s daily Hovmoller consumed monthly means and
    the ``DiurnalCycle`` diagnostic built a "diurnal cycle" out of the two
    times-of-day that appear on a monthly-mean time axis (00:00 and 12:00)
    and compared those 2 points against a real 24-point hourly ERA5 curve.
    """
    direct = sorted(cmor_dir.glob("*.nc"))
    if direct:
        return direct
    frequencies = list(frequencies)
    variable_paths = sorted(
        p
        for table in frequency_cmor_tables(frequencies)
        for p in (cmor_dir / table).glob("*.nc")
    )
    if not variable_paths:
        # No model output at this frequency. Returning the fixed fields alone
        # would let the caller run the group against nothing but areacella /
        # sftlf and report every variable as "missing data"; returning nothing
        # lets it skip the group with an accurate reason instead.
        return []
    fixed_paths = sorted(
        p for table in _FIXED_CMOR_TABLES for p in (cmor_dir / table).glob("*.nc")
    )
    return variable_paths + fixed_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmor-dir", required=True,
                         help="CMOR output root (…/cmor) or a single MIP-table "
                              "directory (…/cmor/Amon).")
    parser.add_argument("--suite", dest="suites", nargs="+", default=[],
                         help="ClimateEval suite names to render into a single "
                              "combined report. Default (empty) = ALL bundled "
                              "suites; a suite missing its data is skipped + "
                              "reported, not fatal.")
    parser.add_argument("--model-id", default="legoESM-1-0")
    parser.add_argument("--experiment-id", default="amip")
    parser.add_argument("--variant-id", default="r1i1p1f1")
    parser.add_argument("--data-root-dir", required=True,
                         help="ClimateEval reference-data root "
                              "(source_id/frequency/var layout).")
    parser.add_argument("--timerange", default="",
                         help="Variable timerange override, e.g. 19790101/19791231. "
                              "Empty = use the model output's own time span.")
    parser.add_argument("--fail-on-missing-data", action="store_true", default=False)
    parser.add_argument("--fail-on-metric-error", action="store_true", default=False,
                         help="ClimateEval's own default (True) turns ANY single "
                              "variable's metric-calculation error (e.g. a "
                              "diagnostic requiring a longer timerange than a "
                              "short run provides) into a MetricError that our "
                              "SUITE-level try/except then catches, discarding "
                              "every OTHER diagnostic in that suite too. Default "
                              "here is False: log the failing variable as a "
                              "warning and keep scoring the rest of the suite.")
    parser.add_argument("--download-missing-data", action="store_true", default=False)
    parser.add_argument("--output-dir", required=True,
                         help="Directory to write per-suite .ddb files and the "
                              "combined HTML report into.")
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME,
                         help="Filename of the combined HTML report written "
                              f"under --output-dir (default: {DEFAULT_REPORT_NAME}).")
    parser.add_argument("--reference-policy",
                         choices=("designated", "era5-only"), default="designated",
                         help="'designated' (default): use each variable's own "
                              "suite reference (CERES-EBAF for TOA fluxes, MERRA2 "
                              "for surface fluxes, GPCP for pr, ...) when it is "
                              "staged, falling back to ERA5 only when it is not. "
                              "'era5-only': force ERA5 everywhere, reproducing the "
                              "older ERA5-only scorecard (still availability-"
                              "checked). NOTE 'designated' changes the reference "
                              "for pr/tas, so those numbers are NOT comparable to "
                              "an era5-only run.")
    parser.add_argument("--run-unscoreable-groups", action="store_true", default=False,
                         help="Run a suite's diagnostics even when "
                              "'climateeval check' finds NO usable model variable "
                              "for them, i.e. when no metric can possibly be "
                              "produced. Off by default because it is expensive: "
                              "on an atmosphere-only AMIP tree this re-enables "
                              "Tier2_ocean_monthly and Tier2_sea_ice_monthly, and "
                              "Tier2_ocean_monthly alone costs ~4 h (ORAS5 mlotst "
                              "area-weighted regridding onto an irregular grid) to "
                              "produce ZERO metrics. What the default costs you is "
                              "reference-only rows (the observed RAPID AMOC series "
                              "and the ORAS5 mixed-layer-depth annual cycle): "
                              "curves with no model to compare against. Pass this "
                              "if you want those reference curves in the report.")
    parser.add_argument("--keep-other-data", action="store_true", default=False,
                         help="Keep 'other_data' (CMIP6 multi-model "
                              "intercomparison) entries. Off by default: the point "
                              "of --evaluate is model-vs-observations, and no CMIP6 "
                              "data is normally staged in a personal data root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Deferred: only the ClimateEval environment has these installed; keeping
    # them out of module scope lets the pure transforms
    # (apply_reference_fallback / group_blocks_by_frequency / cmor_nc_paths /
    # suite_skip_reason / write_suite_yaml / build_arg_parser) be imported and
    # unit-tested from the legoESM env too.
    from importlib import resources

    import iris
    import yaml
    from climateeval.data import DataSourceInformation
    from climateeval.report import serve
    from climateeval.suites import Suite

    cmor_dir = Path(args.cmor_dir)

    cube_cache: dict[tuple[str, ...], Any] = {}

    def cubes_for(frequencies: frozenset[str]) -> Any:
        """Load (once per table set) the model cubes for these frequencies."""
        tables = cmor_tables_for(frequencies)
        if tables not in cube_cache:
            paths = cmor_nc_paths(cmor_dir, frequencies)
            if not paths:
                cube_cache[tables] = None
            else:
                print(
                    f"Loading {len(paths)} CMOR files from {cmor_dir} "
                    f"for frequency {sorted(frequencies)} (tables {list(tables)})",
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    loaded = iris.load([str(p) for p in paths])
                print(f"Loaded {len(loaded)} cubes")
                cube_cache[tables] = loaded
        return cube_cache[tables]

    data_info = DataSourceInformation(
        name=args.model_id,
        category="legoESM",
        institute="MPI-M",
        exp=args.experiment_id,
        variant=args.variant_id,
    )
    diagnostic_kwargs = {
        "data_root_dir": Path(args.data_root_dir),
        "fail_on_missing_data": args.fail_on_missing_data,
        "fail_on_metric_error": args.fail_on_metric_error,
        "download_missing_data": args.download_missing_data,
    }
    variable_kwargs = {"timerange": args.timerange} if args.timerange else {}
    suite_dir = Path(str(resources.files("climateeval.suites")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # No --suite -> run ALL bundled suites (every tier), sorted for stable order.
    suites = args.suites or sorted(p.stem for p in suite_dir.glob("*.yml"))

    is_available = make_availability_checker(Path(args.data_root_dir))

    db_paths: list[Path] = []
    skipped: list[str] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="legoesm_climateeval_"))
    for suite in suites:
        db_path = suite_db_path(output_dir, suite)
        if db_path.is_file():
            db_path.unlink()
        # Hoisted out of the try: if a LATER frequency group raises, an
        # EARLIER group's results are already in the database and must be
        # kept, not discarded with it (same failure mode as the per-variable
        # metric error that used to zero out a whole suite).
        wrote_any = False
        try:
            with (suite_dir / f"{suite}.yml").open() as f:
                raw_suite_def = yaml.safe_load(f)

            inapplicable = suite_skip_reason(suite, args.experiment_id, raw_suite_def)
            if inapplicable:
                skipped.append(f"{suite}: {inapplicable}")
                print(f"SKIPPED suite {suite}: {inapplicable}")
                continue

            suite_def, notes = apply_reference_fallback(
                raw_suite_def,
                is_available,
                era5_only=args.reference_policy == "era5-only",
                strip_other_data=not args.keep_other_data,
            )
            print(f"--- {suite}: reference resolution ({len(notes)} adjustment(s))")
            for note in notes:
                print(f"      {note}")

            groups = group_blocks_by_frequency(suite_def)
            for frequencies, blocks in groups:
                group_cubes = cubes_for(frequencies)
                if group_cubes is None:
                    names = sorted(
                        b.get("name", "?") for b in blocks if isinstance(b, dict)
                    )
                    reason = (
                        f"no model output at frequency {sorted(frequencies)} "
                        f"(tables {list(cmor_tables_for(frequencies))} absent under "
                        f"{cmor_dir}) -> diagnostic(s) {names} not run"
                    )
                    skipped.append(f"{suite}: {reason}")
                    print(f"SKIPPED (partial) {suite}: {reason}")
                    continue

                # One file per group: unique (so a later read can never pick
                # up another group's blocks) and named after the suite, since
                # ClimateEval logs a Suite by its file stem.
                stem = suite if len(groups) == 1 else (
                    f"{suite}__{'-'.join(sorted(frequencies))}"
                )
                group_yml = tmp_dir / f"{stem}.yml"
                write_suite_yaml(blocks, group_yml)
                outcome = check_model_input(
                    group_yml, cmor_dir, args.timerange, frequencies,
                )
                if is_group_unscoreable(outcome.statuses) and (
                    not args.run_unscoreable_groups
                ):
                    names = sorted(
                        b.get("name", "?") for b in blocks if isinstance(b, dict)
                    )
                    reason = (
                        f"no usable model variable at frequency "
                        f"{sorted(frequencies)} "
                        f"({', '.join(sorted(outcome.statuses))} all fail the "
                        f"'climateeval check') -> no metric is possible; "
                        f"diagnostic(s) {names} not run "
                        f"(override with --run-unscoreable-groups)"
                    )
                    skipped.append(f"{suite}: {reason}")
                    print(f"SKIPPED (unscoreable) {suite}: {reason}")
                    continue
                if not outcome.passed:
                    print(
                        f"    NOTE: 'climateeval check' reported unusable input for "
                        f"{suite} at frequency {sorted(frequencies)}; "
                        f"running anyway, affected variables will be skipped.",
                    )
                Suite(
                    group_yml,
                    diagnostic_kwargs=diagnostic_kwargs,
                    variable_kwargs=variable_kwargs,
                ).get_database(
                    group_cubes,
                    data_info,
                    database_resource=f"duckdb://{db_path}",
                    append=wrote_any,
                )
                wrote_any = True
            if not wrote_any:
                db_path.unlink(missing_ok=True)
                continue
        except Exception as exc:  # missing data / inapplicable suite -> skip + report
            skipped.append(f"{suite}: {type(exc).__name__}: {exc}")
            print(f"SKIPPED suite {suite}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not wrote_any:
                db_path.unlink(missing_ok=True)
                continue
            print(
                f"  ...keeping {db_path} with the group(s) that did score "
                f"before the failure.",
            )
        print(f"Database written to {db_path}")
        db_paths.append(db_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if skipped:
        print(f"Skipped/partially skipped {len(skipped)} of {len(suites)} suite(s):")
        for s in skipped:
            print(f"  - {s}")
    if not db_paths:
        print("ERROR: no suite produced a database (all skipped)", file=sys.stderr)
        return 1

    # Single combined report over the suites that scored.
    html_path = output_dir / args.report_name
    serve(db_paths, model_name=args.model_id, save_html=html_path)
    print(f"Report covers {len(db_paths)}/{len(suites)} suites; "
          f"HTML report written to {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
