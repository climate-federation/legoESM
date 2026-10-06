"""Diff a proposed AMIP run's RESOLVED configuration against a previous run's.

The tropical rain regression of August 2026 was ten settings nobody chose: a
new run was built on a different base configuration from the run it was meant
to match, and the difference was invisible because nobody compared what the two
runs RESOLVED to -- only what their decks said.  A deck is a pointer; the
resolved configuration is the fact.

This is that comparison, as a command.  Give it a previous run directory and
the flags of a proposed run, and it prints every field that differs.  Fields
you expect to differ can be named, and anything outside that set is reported as
UNINTENDED and sets a non-zero exit status, so it can gate a launch.

Usage:
    amip_config_diff.py --against <run_dir> --expect land_surface_scheme,... -- <run_amip flags>
    amip_config_diff.py --against <run_dir> --run <other_run_dir> --expect ...

The second form compares two runs that have both started (e.g. the arm and the
control of an A/B after day 1), each from its own recorded configuration.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

# Fields that describe WHERE a run writes and HOW LONG it runs, rather than what
# physics it runs.  Matched by EXACT NAME, never by substring: an earlier version
# skipped anything containing "diag", which silently swallowed
# ``cloud_q_c_diagnostic`` -- a cloud-water threshold that differed by a factor
# of twenty between the two runs being compared.  A substring list cannot
# distinguish a diagnostic CADENCE from a physics field whose name happens to
# contain the same letters, so it must not be used here.
_SKIP_EXACT = frozenset({
    "output_dir", "run_name", "days", "start_day", "start_year", "seed",
    "checkpoint_days", "diag_days", "restart_from", "config", "params",
    "ic_path", "forcing_path", "sic_path", "solar_file", "ozone_file",
    "ghg_file", "aerosol_file", "volcanic_aerosol_file", "topography",
    "surfdata_path", "clm_surfdata_path", "land_mask_path", "land_mask_file",
    "subgrid_orography_path", "subgrid_orography_file",
    "land_ic_path", "max_wallclock_seconds", "output", "cmor_output",
    "monthly_means",
})


def _skip(key: str) -> bool:
    """True for fields that say where a run writes, not what it computes."""
    return key.split(".")[-1] in _SKIP_EXACT


def _as_plain(v):
    """NamedTuples -> dicts, all the way down.

    A run's recorded configuration is nested JSON, while a freshly built one is
    nested NamedTuples.  Comparing the two without this turns every nested block
    into one opaque tuple-vs-dict difference and buries the real ones.
    """
    if hasattr(v, "_asdict"):
        return {k: _as_plain(x) for k, x in v._asdict().items()}
    if isinstance(v, dict):
        return {k: _as_plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_as_plain(x) for x in v]
    return v


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _resolve(flags, repo):
    """Build the ExperimentConfig the given flags would produce."""
    spec = importlib.util.spec_from_file_location("_ra", repo / "scripts/run/run_amip.py")
    ra = importlib.util.module_from_spec(spec)
    sys.modules["_ra"] = ra
    spec.loader.exec_module(ra)
    from legoesm.driver.run_config_yaml import load_yaml_config

    parser = ra.build_arg_parser()
    pre, _ = parser.parse_known_args(flags)
    if pre.config is not None:
        keys = load_yaml_config(pre.config, parser, example_keys="")
        parser.set_defaults(**keys)
        parser.set_defaults(_config_keys=frozenset(keys))
    args = parser.parse_args(flags)
    args = ra._postprocess_args(args, parser, flags)
    config = ra.build_config_from_args(args)

    # Apply --params exactly as the run script does, AFTER the config is built.
    # Without this the tuned convection and microphysics values silently read as
    # their defaults and the diff invents differences that the real run does not
    # have -- the instrument would then be manufacturing the very finding it
    # exists to detect.
    if getattr(args, "params", None):
        from legoesm.driver.run_config_yaml import (
            load_params_config, build_atm_scalar_param_map,
            apply_params_to_config)
        scalar_map = build_atm_scalar_param_map()
        mapped = {q: v for q, v in load_params_config(args.params).items()
                  if q in scalar_map}
        if mapped:
            config = apply_params_to_config(config, mapped, driver="run_amip",
                                            scalar_param_map=scalar_map)
    return config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--against", required=True,
                    help="previous run directory holding experiment_config.json")
    ap.add_argument("--expect", default="",
                    help="comma-separated fields that are MEANT to differ")
    ap.add_argument("--repo", default=".", help="repository root")
    ap.add_argument("--run", default=None,
                    help="a second run directory holding experiment_config.json, "
                         "compared instead of resolving flags")
    ap.add_argument("flags", nargs=argparse.REMAINDER,
                    help="the proposed run's run_amip flags, after --")
    args = ap.parse_args(argv)

    flags = [f for f in args.flags if f != "--"]
    repo = pathlib.Path(args.repo).resolve()
    ref = _flatten(json.loads(
        (pathlib.Path(args.against) / "experiment_config.json").read_text()))
    if args.run is not None:
        if flags:
            ap.error("give either --run or run_amip flags, not both")
        new = _flatten(json.loads(
            (pathlib.Path(args.run) / "experiment_config.json").read_text()))
    else:
        new = _flatten(_as_plain(_resolve(flags, repo)))

    expected = {e.strip() for e in args.expect.split(",") if e.strip()}
    intended, unintended = [], []
    for key in sorted(set(ref) | set(new)):
        if _skip(key):
            continue
        a, b = ref.get(key, "<absent>"), new.get(key, "<absent>")
        if str(a) == str(b):
            continue
        (intended if key.split(".")[-1] in expected else unintended).append((key, a, b))

    label = pathlib.Path(args.against).name
    if intended:
        print(f"INTENDED differences vs {label} ({len(intended)}):")
        for k, a, b in intended:
            print(f"  {k:38s} {str(a)[:20]:>22} -> {str(b)[:20]}")
    if unintended:
        print(f"\nUNINTENDED differences vs {label} ({len(unintended)}) "
              f"-- each one is a setting nobody chose:")
        for k, a, b in unintended:
            print(f"  {k:38s} {str(a)[:20]:>22} -> {str(b)[:20]}")
        print("\nEither pin these to the reference, or add them to --expect "
              "with a reason. A run launched over this diff is not a controlled "
              "comparison.")
        return 1
    print(f"\nNo unintended differences vs {label}: the proposed run changes "
          f"only what was asked for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
