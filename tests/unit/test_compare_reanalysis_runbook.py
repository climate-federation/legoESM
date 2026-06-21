"""Doc-code consistency guard for the operator runbook.

``docs/compare_reanalysis_runbook.md`` gives an HPC operator the exact commands to
run the empirical demonstration.  A command that silently drifts out from under the
doc (a script renamed/moved, a flag removed) is worse than no runbook -- the operator
burns cluster time on a broken command.  These tests make the iter-237 manual
verification PERMANENT: every script the runbook cites exists, and every ``--flag`` it
documents is real.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]              # tests/unit/ -> repo root
_RUNBOOK = _REPO / "docs" / "compare_reanalysis_runbook.md"
# The campaign CLI the .sbatch launcher wraps: the runbook documents its knobs
# (--feedback-strategy, --iterations, …) even though it is invoked via sbatch, so its
# source is part of the flag-search set.
_CAMPAIGN_CLI = "scripts/run/run_correction_campaign.py"

_SCRIPT_RE = re.compile(r"scripts/[A-Za-z0-9_/]+\.(?:py|sbatch)")
_FLAG_RE = re.compile(r"--[a-z][a-z0-9-]+")


def _runbook_text() -> str:
    return _RUNBOOK.read_text()


def _referenced_scripts(text: str) -> list[str]:
    return sorted(set(_SCRIPT_RE.findall(text)))


def test_runbook_exists():
    assert _RUNBOOK.is_file(), f"missing operator runbook at {_RUNBOOK}"


def test_runbook_referenced_scripts_all_exist():
    paths = _referenced_scripts(_runbook_text())
    assert paths, "no script paths parsed from the runbook (regex/file broken?)"
    missing = [p for p in paths if not (_REPO / p).is_file()]
    assert not missing, f"runbook cites nonexistent script(s): {missing}"


def test_runbook_documented_flags_are_real():
    text = _runbook_text()
    scripts = _referenced_scripts(text)
    scripts.append(_CAMPAIGN_CLI)                        # the campaign CLI the launcher wraps
    sources = "\n".join(
        (_REPO / s).read_text() for s in scripts if (_REPO / s).is_file())
    flags = sorted(set(_FLAG_RE.findall(text)))
    assert flags, "no --flags parsed from the runbook"
    # Each documented flag must appear (as an argparse option or sbatch arg) in at
    # least one referenced script's source -- else the doc has drifted / has a typo.
    unreal = [f for f in flags if f not in sources]
    assert not unreal, (
        f"runbook documents flag(s) absent from every referenced script: {unreal}")


def test_runbook_distributed_section_symbols_are_importable():
    """§7 (distributed / HPC) documents a Python code block calling specific helpers; a
    rename would silently break the operator's copy-pasted driver.  Pin that every cited
    symbol still imports from where the runbook imports it (iter 263)."""
    text = _runbook_text()
    assert "## 7. Distributed / HPC-scale" in text, "runbook lost its distributed section"
    # the §7 block uses return_layout=True on the entry point — pin that kwarg exists.
    import inspect

    from legoesm.training.campaign_summary import (  # noqa: F401
        campaign_health,
        summarize_campaign,
    )
    from legoesm.training.distributed_campaign import (  # noqa: F401
        assemble_global_campaign_result,
    )

    from scripts.run.run_correction_campaign import (  # noqa: F401
        build_campaign_output_dict,
        build_distributed_mpas_campaign,
    )

    sig = inspect.signature(build_distributed_mpas_campaign)
    assert "return_layout" in sig.parameters, "runbook §7 uses return_layout= (now missing)"
    # and the symbols the block calls by name must actually appear in the doc text.
    for name in ("build_distributed_mpas_campaign", "assemble_global_campaign_result",
                 "build_campaign_output_dict", "return_layout"):
        assert name in text, f"runbook §7 no longer mentions {name}"


def test_runbook_automated_gating_clis_are_exit_code_gateable():
    """The 'Automated gating' section (iter 290) promises every workflow CLI exits 0 only on
    success so the pipeline chains with ``&&``.  Pin that contract: each of the 5 cited CLIs
    exposes a ``main(argv)`` entry point (the gateable surface) — a regression removing one,
    or the runbook citing a non-CLI, breaks the documented automation."""
    import importlib

    text = _runbook_text()
    assert "## Automated gating (exit codes)" in text, "runbook lost the gating section"
    clis = {
        "scripts.experiment.smoke_compare_reanalysis",
        "scripts.validate.run_perfect_model_osse",
        "scripts.run.run_correction_campaign",
        "scripts.experiment.check_campaign_deploy",
        "scripts.validate.compare_amip_era5",
    }
    for mod_name in clis:
        # the CLI basename must be cited in the gating section (the table uses basenames;
        # the campaign is invoked via its .sbatch wrapper) AND expose a callable main(argv).
        basename = mod_name.rsplit(".", 1)[-1] + ".py"
        assert basename in text, f"runbook gating section no longer cites {basename}"
        main = getattr(importlib.import_module(mod_name), "main", None)
        assert callable(main), f"{mod_name} lost its gateable main()"


def test_campaign_sbatch_flags_are_valid_cli_options():
    """The production SLURM launcher passes campaign-CLI flags (--config, --era5-zarr,
    --mode, --diagnosis-method, --iterations, --n-worst, --les-budget, --orographic-forcing,
    --out, --coupled-preset, --dry-run, --checkpoint, --resume) to run_correction_campaign.py.
    A flag RENAMED/REMOVED in the CLI would break the production launch at argparse with NO
    test catching it — the runbook test above pins the RUNBOOK's flags, not the sbatch's own.
    Lock the sbatch↔CLI contract.  SLURM directives (#SBATCH) and the config-gen comment
    examples are excluded (comment lines start with #) (iter 308)."""
    from scripts.run.run_correction_campaign import _build_arg_parser

    sbatch = _REPO / "scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch"
    assert sbatch.is_file(), f"missing campaign launcher at {sbatch}"
    # Campaign flags = --xxx on NON-comment lines (excludes #SBATCH SLURM directives + the
    # config-generator comment examples — both of which begin with #).
    campaign_flags: set[str] = set()
    for line in sbatch.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        campaign_flags.update(_FLAG_RE.findall(line))
    assert campaign_flags, "no campaign flags parsed from the sbatch (regex/file broken?)"

    valid = {opt for action in _build_arg_parser()._actions for opt in action.option_strings}
    unknown = sorted(f for f in campaign_flags if f not in valid)
    assert not unknown, (
        f"the campaign sbatch passes flag(s) the CLI does not define: {unknown} — a "
        "renamed/removed CLI option would break the production launch at argparse.")


def test_preflight_sbatch_flags_are_valid_cli_options():
    """The REALISTIC-pre-flight SLURM wrapper (iter 443: rrtmgp is compute-node-only) passes
    ck_sensitivity-CLI flags; a renamed/removed CLI option would break the batch pre-flight at
    argparse. Lock the sbatch↔CLI contract (mirrors the campaign test). #SBATCH directives +
    comment examples (--days-sweep, sbatch --dependency) are excluded (they start with #)."""
    from scripts.experiment.ck_sensitivity_vs_era5 import _build_arg_parser as _ck_parser

    sbatch = _REPO / "scripts/cluster/compare_reanalysis/preflight_ck_sensitivity.sbatch"
    assert sbatch.is_file(), f"missing pre-flight launcher at {sbatch}"
    flags: set[str] = set()
    for line in sbatch.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        flags.update(_FLAG_RE.findall(line))
    assert flags, "no flags parsed from the pre-flight sbatch (regex/file broken?)"
    valid = {opt for action in _ck_parser()._actions for opt in action.option_strings}
    unknown = sorted(f for f in flags if f not in valid)
    assert not unknown, (
        f"the pre-flight sbatch passes flag(s) the ck_sensitivity CLI does not define: "
        f"{unknown} — a renamed/removed option would break the batch pre-flight at argparse.")


def test_osse_preflight_sbatch_flags_are_valid_cli_options():
    """The PERFECT-MODEL OSSE go/no-go SLURM wrapper (iter 491: the twin runs the model
    multiple times -> compute-node batch job) passes run_perfect_model_osse-CLI flags; a
    renamed/removed CLI option would break the batch pre-flight at argparse. Locks the
    sbatch↔CLI contract (mirrors the ck-sensitivity + campaign tests)."""
    from scripts.validate.run_perfect_model_osse import _build_argparser as _osse_parser

    sbatch = _REPO / "scripts/cluster/compare_reanalysis/preflight_osse.sbatch"
    assert sbatch.is_file(), f"missing OSSE pre-flight launcher at {sbatch}"
    flags: set[str] = set()
    for line in sbatch.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        flags.update(_FLAG_RE.findall(line))
    assert flags, "no flags parsed from the OSSE pre-flight sbatch (regex/file broken?)"
    valid = {opt for action in _osse_parser()._actions for opt in action.option_strings}
    unknown = sorted(f for f in flags if f not in valid)
    assert not unknown, (
        f"the OSSE pre-flight sbatch passes flag(s) the run_perfect_model_osse CLI does not "
        f"define: {unknown} — a renamed/removed option would break the batch pre-flight.")


# Detect a top-level OR function-scope ``from scripts.* import`` / ``import scripts.*``.
_IMPORTS_SCRIPTS_RE = re.compile(r"^\s*(?:from\s+scripts[.\s]|import\s+scripts\b)", re.M)


def test_runbook_workflow_scripts_run_as_scripts_have_the_path_bootstrap():
    """Static audit-as-a-test for the iter-321 launch bug class, comprehensively.

    A runbook command run AS A SCRIPT (``python scripts/X/Y.py``, the sbatch/runbook way)
    puts the script's OWN directory on ``sys.path`` — NOT the repo root — so any ``from
    scripts.* import`` raises ``ModuleNotFoundError: No module named 'scripts'`` at launch
    unless the file adds the repo root to ``sys.path`` first.  iter 321 found the campaign +
    OSSE were missing the bootstrap that smoke + deploy-check already had; the subprocess test
    in ``test_cli_imports_resolve`` confirms it at runtime for the 4 main CLIs.  This locks the
    invariant STATICALLY for EVERY workflow ``.py`` the runbook cites (plus the campaign the
    .sbatch wraps): a file importing ``scripts.*`` MUST carry a ``sys.path.insert`` bootstrap.
    Future-proofs a NEW runbook script (e.g. a plotter that later imports ``scripts.*``) —
    the subprocess test's curated list would not cover it; this audit does."""
    scripts = set(_referenced_scripts(_runbook_text()))
    scripts.add(_CAMPAIGN_CLI)                       # wrapped by the .sbatch, not a direct .py ref
    py_scripts = sorted(s for s in scripts if s.endswith(".py"))
    assert py_scripts, "no .py workflow scripts parsed from the runbook (regex/file broken?)"

    offenders = []
    for rel in py_scripts:
        src = (_REPO / rel).read_text()
        if _IMPORTS_SCRIPTS_RE.search(src) and "sys.path.insert" not in src:
            offenders.append(rel)
    assert not offenders, (
        f"workflow script(s) import `scripts.*` but lack a sys.path bootstrap: {offenders} — "
        "they raise `ModuleNotFoundError: No module named 'scripts'` when run as a script (the "
        "documented invocation).  Add before the `from scripts...` import:\n"
        "    if str(Path(__file__).resolve().parents[2]) not in sys.path:\n"
        "        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))\n"
        "(cf. iter 321).")


def test_path_bootstrap_audit_is_non_vacuous():
    """Non-vacuity: the audit's predicate (imports-scripts AND no-bootstrap) actually FIRES
    on the iter-321 failure shape, so a green result means the invariant holds — not that the
    check never triggers.  Synthetic sources, no file I/O."""
    bug = "from scripts.run.run_correction_campaign import x\nprint('hi')\n"
    fixed = ("import sys\nfrom pathlib import Path\n"
             "sys.path.insert(0, str(Path(__file__).resolve().parents[2]))\n"
             "from scripts.run.run_correction_campaign import x\n")
    clean = "import numpy as np\nprint('no scripts import here')\n"
    assert _IMPORTS_SCRIPTS_RE.search(bug) and "sys.path.insert" not in bug      # flagged
    assert _IMPORTS_SCRIPTS_RE.search(fixed) and "sys.path.insert" in fixed       # passes
    assert not _IMPORTS_SCRIPTS_RE.search(clean)                                  # not applicable


def test_runbook_documents_the_starved_correction_diagnostics():
    """The runbook must teach the operator to read the diagnostics the iter-508..524 arc added
    (iter 526): the realism breakdown, the diagnosis valid-level count, and the
    C_K-insensitivity/rrtmgp guidance — else those operator-facing outputs are produced but the
    operator does not know they exist or what they mean. A doc-consistency tripwire: removing the
    §3b diagnose-a-stalled-correction guidance fails here."""
    text = _runbook_text()
    # the three rejection causes a no-go verdict can have, each surfaced by the run:
    assert "LES realism:" in text                       # the realism breakdown line (iter 512/520)
    assert "valid diagnosis levels" in text             # the diagnosis-validity count (iter 524)
    assert "C_K-INSENSITIVE" in text and "rrtmgp" in text  # the idealized-radiation guidance (515)
    assert "n_diagnoses_valid" in text                  # the column-count signal (iter 100/510)
