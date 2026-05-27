"""Regression tests for the plane CRM 30-day wrapper script defaults
and post-run summarizer failure propagation behaviour.

Locks the iter-14 production-measured config as a contract on
``scripts/run_rce_30day.sh`` env-var defaults. iter-58 found this
wrapper still had ``DT=1.0`` + ``N_ACOUSTIC=24`` in its
documentation (citing the iter-1 F1 ladder pre-F10 fix), while the
iter-14 production measurement + iter-38 structural regression
both use ``DT=5.0`` + ``N_ACOUSTIC=12``.

This test parses the bash-script env-var defaults via a simple
regex and asserts they match the iter-14 / iter-38 production
contract. A future edit that silently reverts the defaults trips
this in < 1 s before any of the slow nightly tests would catch
the drift.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
WRAPPER = REPO_ROOT / "scripts" / "run_rce_30day.sh"


def _parse_env_default(text: str, name: str) -> str | None:
    """Extract the ``${X:-VAL}`` default for env-var ``X`` from a
    bash assignment like ``X="${X:-VAL}"``. Returns the unquoted
    default string or ``None`` if no assignment found."""
    m = re.search(
        rf'^\s*{re.escape(name)}\s*=\s*"\$\{{{re.escape(name)}:-([^}}]+)\}}"\s*$',
        text, re.MULTILINE,
    )
    if m is None:
        return None
    return m.group(1).strip()


@pytest.fixture(scope="module")
def wrapper_text() -> str:
    assert WRAPPER.exists(), f"wrapper script missing: {WRAPPER}"
    return WRAPPER.read_text()


def test_dt_default_matches_iter180_production(wrapper_text):
    """iter-180 refresh: WENO5 horizontal advection + stronger
    acoustic off-centering (beta=0.2) lifts the iter-14 dt=5 s
    contract to dt=20 s. Smoke measurements at 32x32x30 dx=4 km
    with WENO5 + beta=0.2 + radiation showed stable 432-step runs
    at max|w|<=1.8e-3 m/s and Ca_substep=0.87 (well under the
    SI-relaxed acoustic CFL bound). 4x speedup over iter-14.

    Pre iter-180 the wrapper defaulted DT=5.0 (the iter-14 measured
    value with upwind1 + beta=0.1). The new combination is now the
    production contract.
    """
    dt = _parse_env_default(wrapper_text, "DT")
    assert dt == "20.0", (
        f"run_rce_30day.sh DT default = {dt!r}, expected '20.0' "
        f"(iter-180 WENO5 + beta=0.2 production contract)."
    )


def test_n_acoustic_default_matches_iter14_production(wrapper_text):
    """iter-14 + iter-38 measure N_ACOUSTIC=12 for dt=5.0 s.
    The wrapper must default to that value; iter-58 found it
    stale at 24 (the iter-1 dt=1.0 s value)."""
    n_ac = _parse_env_default(wrapper_text, "N_ACOUSTIC")
    assert n_ac == "12", (
        f"run_rce_30day.sh N_ACOUSTIC default = {n_ac!r}, "
        f"expected '12' (iter-14 + iter-38 production contract). "
        f"With dt=5.0 + N_ACOUSTIC=24 the acoustic CFL ratio "
        f"would halve, producing a slower / over-stable substep."
    )


def test_other_production_defaults(wrapper_text):
    """Lock the remaining production defaults so a partial revert
    surfaces the same way."""
    defaults = {
        "NX": "132",
        "NY": "132",
        # iter-180: WENO5 is the production advection scheme (5th-order,
        # less grid-scale dispersion than upwind1; pairs with dt=20 + beta=0.2
        # at acoustic CFL Ca_sub=0.87 in the iter-180 smoke).
        "ADVECTION": "weno5",
        "HYPERDIFF": "5.0e6",
        "BUBBLE_K": "0.0",   # F7 / F10 contract: clean Wing IC
        # iter-179: QV_NOISE flipped from 0.0 -> 1e-4 to break the
        # iter-149 column-symmetric convection trap (Wing IC + no
        # bubble + no qv noise + mean-wind removal made every column
        # evolve identically → no horizontal organization → 1500x
        # below Wing 2018 precip target). 1e-4 is 50x smaller than
        # the iter-97 F7 destabilising 5e-3.
        "QV_NOISE": "1e-4",
        # iter-61 Codex MEDIUM coverage gap fix: env defaults that
        # iter-58 missed.
        "DAYS": "30",
        "RANKS": "12",
        "USE_DD": "0",   # legacy rank-0-broadcast is the F8-stable default
        # iter-64: PYBIN was the last env default uncovered. A revert
        # to a non-venv Python would silently break the pinned
        # JAX/mpi4jax/JAX-MPI versions iter-10 / iter-11 stacked.
        "PYBIN": ".venv/bin/python",
        # iter-95g: 30-day RCE wrapper defaults to --no-mass-fixer.
        # The legacy fix_moist_mass_plane rescales total water back to
        # IC every step which kills RCE spinup (surface flux must NET
        # ADD moisture until precip balances). The 30-day wrapper is
        # RCE-specific, so default ON (NO_MASS_FIXER=1). Gravity-wave
        # smokes that want the fixer can flip it to 0.
        "NO_MASS_FIXER": "1",
        # iter-103: EVALUATE_DOD=0 default keeps 1-day smokes / 0.05-day
        # sanity runs non-gated. Production 30-day runs are expected to
        # set EVALUATE_DOD=1 explicitly (the wrapper docstring + the
        # test_wrapper_evaluate_dod_threads_into_summarizer test below
        # both spell this out).
        "EVALUATE_DOD": "0",
    }
    for name, expected in defaults.items():
        actual = _parse_env_default(wrapper_text, name)
        assert actual == expected, (
            f"run_rce_30day.sh {name} default = {actual!r}, "
            f"expected {expected!r} (iter-12/14/38 production contract)."
        )


# iter-61 Codex MEDIUM silent-pass fix: the wrapper passes several
# production-relevant flags HARDCODED in the mpirun argv (not via
# env vars). A partial revert of these would slip past both this
# test + the driver-defaults test. Lock them here.


_HARDCODED_PASSTHROUGH_FLAGS = {
    # iter-180 refresh: beta=0.1 (iter-14/38 dt=5) -> 0.2 to pair with
    # the dt=20 + WENO5 production combo. Stronger off-centering
    # damps the acoustic mode without needing more substeps.
    "--acoustic-off-centering": "0.2",
    "--snapshot-hours": "24.0",          # daily snapshots
    "--snapshot-3d-hours": "1.0",        # hourly 3D snapshots
    "--profile-days": "5.0",
    "--log-every-steps": "100",
    # --semi-implicit-acoustic is a store_true; verified separately.
}


@pytest.mark.parametrize(
    "flag,expected_value",
    sorted(_HARDCODED_PASSTHROUGH_FLAGS.items()),
)
def test_wrapper_hardcoded_driver_flag(wrapper_text, flag, expected_value):
    """Each ``flag VALUE`` pair appears verbatim in the wrapper's
    mpirun invocation line (no env-var indirection). Catches a
    silent revert of any production tunable that the iter-58
    env-only coverage missed.
    """
    # The wrapper uses bash backslash-continuation; flags appear on
    # separate continuation lines like
    #   --acoustic-off-centering 0.1 \
    # Match flag then whitespace then the literal value.
    pattern = re.escape(flag) + r"\s+" + re.escape(expected_value) + r"\b"
    assert re.search(pattern, wrapper_text), (
        f"run_rce_30day.sh missing hardcoded {flag} {expected_value} "
        f"in the mpirun invocation. A partial revert of this "
        f"production tunable would slip past both the env-var "
        f"defaults test (iter-58) and the driver argparse test "
        f"(iter-59)."
    )


def test_wrapper_semi_implicit_acoustic_present(wrapper_text):
    """--semi-implicit-acoustic is a bare flag (no value); verify it
    appears in the mpirun invocation. Production contract since
    iter-1 + verified by iter-14/38."""
    assert re.search(r"\\\n\s*--semi-implicit-acoustic\b", wrapper_text), (
        "run_rce_30day.sh missing --semi-implicit-acoustic flag in "
        "mpirun invocation. The SI substep is the iter-14/38 "
        "production contract; removing it falls back to explicit "
        "forward-Euler which is dt-stability-bounded at the iter-1 "
        "ladder (dt <= 1.0 s)."
    )


def test_wrapper_no_mass_fixer_conditional_present(wrapper_text):
    """iter-95g: --no-mass-fixer must be passed when NO_MASS_FIXER=1
    (the wrapper's default). Verify the conditional bash logic is
    intact (NMF_FLAG variable assigned and threaded into the mpirun
    invocation)."""
    # The wrapper assigns NMF_FLAG="--no-mass-fixer" when
    # NO_MASS_FIXER=1, then includes $NMF_FLAG in the mpirun argv.
    assert re.search(
        r'NMF_FLAG\s*=\s*"--no-mass-fixer"', wrapper_text,
    ), (
        "run_rce_30day.sh missing NMF_FLAG=\"--no-mass-fixer\" "
        "assignment. iter-95b's RCE-spinup fix is opt-in via this "
        "driver flag; without it CWV pins at IC and 30-day RCE "
        "never spins up."
    )
    assert re.search(r"\\\n\s*\$NMF_FLAG\b", wrapper_text), (
        "run_rce_30day.sh assigns NMF_FLAG but does not pass "
        "$NMF_FLAG into the mpirun invocation. The flag would "
        "be silently dropped and Bug 2 from iter-95 would return."
    )


def test_wrapper_no_mass_fixer_actually_conditional(wrapper_text):
    """iter-95m (Codex iter-95g-k LOW#2): the iter-95g sibling
    test above only checks text presence — if the bash conditional
    became unconditional (e.g. ``NMF_FLAG="--no-mass-fixer"`` not
    guarded by ``if [ "$NO_MASS_FIXER" = "1" ]``), it would still
    pass. Catch that drift by asserting the wrapper carries the
    guard literally."""
    # Match: NMF_FLAG="" assignment + the bash if guard around the
    # NMF_FLAG="--no-mass-fixer" reassignment.
    assert re.search(r'NMF_FLAG\s*=\s*""\s*\n', wrapper_text), (
        "run_rce_30day.sh missing the empty-default "
        "``NMF_FLAG=\"\"`` initialisation. Without it, "
        "NO_MASS_FIXER=0 (legacy fixer ON) would still pass "
        "--no-mass-fixer to the driver, breaking gravity-wave "
        "smokes that rely on the fixer."
    )
    assert re.search(
        r'if\s*\[\s*"\$NO_MASS_FIXER"\s*=\s*"1"\s*\]\s*;\s*then\s*'
        r'\n\s*NMF_FLAG\s*=\s*"--no-mass-fixer"',
        wrapper_text,
    ), (
        "run_rce_30day.sh missing the bash conditional gate "
        "``if [ \"$NO_MASS_FIXER\" = \"1\" ]; then NMF_FLAG=\"...\"``. "
        "If the assignment is unconditional, NO_MASS_FIXER=0 stops "
        "working and there is no escape hatch back to the legacy "
        "fixer behaviour for gravity-wave / hydrostatic smokes."
    )


def _strip_bash_comments(text: str) -> str:
    """Return ``text`` with bash comment lines removed. A line is a
    comment when its first non-whitespace character is ``#``. iter-99
    Codex MEDIUM#4 fix: regression tests below search for the
    summarizer / mpirun invocations against this stripped view so a
    comment that happens to contain the same tokens cannot satisfy
    the contract."""
    return "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )


def test_wrapper_invokes_post_run_summarizer(wrapper_text):
    """iter-99: ``run_rce_30day.sh`` must call
    ``scripts/summarize_rce_trajectory.py`` AFTER the mpirun line so
    every production run emits ``<OUTPUT>/trajectory.csv``. Three
    invariants (iter-99 Codex review hardened):

    1. The mpirun line MUST NOT be prefixed by ``exec`` — ``exec``
       replaces the shell process and would silently skip every
       later command (the iter-99 mistake we are guarding against).
    2. The summarizer call MUST appear AFTER the mpirun line, point
       at the same ``$OUTPUT`` directory, and use the same
       ``$PYBIN`` (no second venv drift). Matched against the
       comment-stripped view (Codex MEDIUM#4).
    3. The summarizer line MUST start at the beginning of an
       executable shell line (no leading non-whitespace), so a
       commented-out invocation cannot satisfy the contract.
    """
    code = _strip_bash_comments(wrapper_text)
    # Invariant 1: no ``exec`` before the mpirun line in CODE (not
    # comments — the wrapper header documents the pre-iter-99 form).
    assert not re.search(
        r"^\s*exec\s+mpirun\b", code, re.MULTILINE,
    ), (
        "run_rce_30day.sh uses ``exec mpirun`` — that replaces the "
        "shell process and skips the post-run summarizer below. "
        "Drop the ``exec``; pipefail (already set) preserves "
        "mpirun's exit status through ``| tee``."
    )
    # Invariant 2: locate both lines in CODE and verify ordering.
    mpi_match = re.search(
        r"^\s*mpirun\s+-np\s+\"?\$RANKS\"?\b",
        code,
        re.MULTILINE,
    )
    # Invariant 3: anchor the summarizer call to ``^\s*"$PYBIN"`` so
    # the line is executable, not a comment-line substring (Codex
    # MEDIUM#4 — comments stripped above, but anchoring belt-and-
    # braces against an executable heredoc / dead branch).
    summary_match = re.search(
        r'^\s*"\$PYBIN"\s+"\$REPO_ROOT/scripts/summarize_rce_trajectory\.py"\s+"\$OUTPUT"',
        code,
        re.MULTILINE,
    )
    assert mpi_match is not None, (
        "run_rce_30day.sh missing the ``mpirun -np \"$RANKS\" ...`` "
        "line in executable code. The wrapper structure changed; "
        "iter-99 post-run hook needs to be re-anchored."
    )
    assert summary_match is not None, (
        "run_rce_30day.sh missing the post-run "
        "``\"$PYBIN\" \"$REPO_ROOT/scripts/summarize_rce_trajectory.py\" "
        "\"$OUTPUT\"`` call in executable code (only matched in "
        "comments?). Production runs would land snapshots but no "
        "aggregated per-day trajectory CSV (iter-99 contract)."
    )
    assert summary_match.start() > mpi_match.start(), (
        "run_rce_30day.sh calls summarize_rce_trajectory.py BEFORE "
        "the mpirun line — the snapshots/ directory would be empty "
        "at that point. Move the summarizer call below the mpirun "
        "pipeline."
    )


def test_wrapper_summarizer_failure_propagates(wrapper_text):
    """iter-99 Codex MEDIUM#1 fix: a successful mpirun followed by a
    failed summarizer must propagate as a non-zero wrapper exit
    (unless explicitly downgraded via ``ALLOW_SUMMARY_FAILURE=1``).
    Lock the three pieces of bash that implement this:

    1. ``ALLOW_SUMMARY_FAILURE`` env default initialisation.
    2. The summarizer call wrapped in ``set +e`` … ``set -e`` so
       ``set -e`` does NOT auto-abort before we inspect the status.
    3. An ``exit "$summary_status"`` on the failing branch when
       ``ALLOW_SUMMARY_FAILURE != 1``.
    """
    code = _strip_bash_comments(wrapper_text)
    assert re.search(
        r'ALLOW_SUMMARY_FAILURE\s*=\s*"\$\{ALLOW_SUMMARY_FAILURE:-0\}"',
        code,
    ), (
        "run_rce_30day.sh missing ALLOW_SUMMARY_FAILURE default. "
        "Without it, the failure-propagation contract has no escape "
        "hatch for runs where the summarizer is expected to fail "
        "(e.g. mpirun aborted before any snapshot landed)."
    )
    assert re.search(r"^\s*set\s+\+e\s*$", code, re.MULTILINE), (
        "run_rce_30day.sh missing ``set +e`` before the summarizer "
        "invocation. Without it, ``set -e`` would kill the shell "
        "the moment the summarizer fails, BEFORE the conditional "
        "branch can decide whether to propagate the failure."
    )
    assert re.search(
        r'^\s*summary_status\s*=\s*\$\?\s*$', code, re.MULTILINE,
    ), (
        "run_rce_30day.sh missing ``summary_status=$?`` capture "
        "after the summarizer call. The fail-vs-warn branch can "
        "only fire if the exit status is recorded before set -e is "
        "re-enabled."
    )
    assert re.search(r'^\s*exit\s+"\$summary_status"', code, re.MULTILINE), (
        "run_rce_30day.sh missing ``exit \"$summary_status\"`` on "
        "the summarizer-failure / ALLOW_SUMMARY_FAILURE != 1 "
        "branch. Without it, the wrapper exits 0 when the summarizer "
        "fails — the exact iter-99 Codex MEDIUM#1 finding."
    )


# iter-101 (Codex iter-100 LOW): behavioural regression test that
# actually executes the wrapper bash with stubbed mpirun + summarizer
# and asserts the exit-status contract end-to-end. The text-regex
# tests above prove the right tokens are present; this one proves the
# wrapper actually does the right thing when the summarizer fails —
# closes the "tokens in wrong order" coverage gap Codex flagged.

import os
import stat
import subprocess


def _make_stub(path: Path, exit_code: int = 0, body: str = "") -> None:
    """Create an executable stub script at ``path`` that prints its
    argv on stdout then exits with ``exit_code``."""
    path.write_text(
        "#!/usr/bin/env bash\n"
        f"echo STUB:{path.name}:\"$@\"\n"
        f"{body}\n"
        f"exit {exit_code}\n"
    )
    path.chmod(
        path.stat().st_mode
        | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )


def _run_wrapper_with_stubs(
    tmp_path: Path,
    *,
    mpirun_exit: int,
    summarizer_exit: int,
    allow_summary_failure: str = "0",
    days: str = "0",
) -> subprocess.CompletedProcess[str]:
    """Run ``scripts/run_rce_30day.sh`` with stubbed mpirun + Python
    interpreter so the wrapper exercises its post-run branch logic
    against synthetic exit codes.

    The stubbed PYBIN is set to point at a tiny bash wrapper whose
    only job is to ``exit summarizer_exit`` so we never invoke the
    real summarizer (which would FileNotFoundError on an empty
    output dir anyway — that exit code is ``1``, which the test
    treats as a real failure, not a behavioural one).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Stub mpirun.
    _make_stub(bin_dir / "mpirun", exit_code=mpirun_exit)
    # Stub the PYBIN — the wrapper calls "$PYBIN" "$REPO_ROOT/.../summarize_rce_trajectory.py" "$OUTPUT".
    # Make PYBIN a script that ignores argv and just exits with the
    # configured code. Use a unique filename so we can point PYBIN at
    # the absolute path (bypasses PATH).
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=summarizer_exit)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    # Prepend bin_dir so ``mpirun`` resolves to our stub.
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = days
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = allow_summary_failure
    # Provide an explicit output dir so the wrapper doesn't pollute
    # results/.
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_wrapper_mpirun_ok_summarizer_ok_exits_zero(tmp_path):
    """Sanity baseline: both stages succeed → exit 0."""
    res = _run_wrapper_with_stubs(
        tmp_path, mpirun_exit=0, summarizer_exit=0,
    )
    assert res.returncode == 0, (
        f"baseline failed: stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert "Wrote " in res.stdout


def test_wrapper_mpirun_fail_propagates(tmp_path):
    """mpirun nonzero → wrapper nonzero (set -e + pipefail). The
    summarizer should NOT even run."""
    res = _run_wrapper_with_stubs(
        tmp_path, mpirun_exit=42, summarizer_exit=0,
    )
    assert res.returncode != 0, (
        f"mpirun failure was masked: {res.stdout!r} {res.stderr!r}"
    )
    # "Computing per-day RCE trajectory summary..." prints AFTER
    # mpirun completes. set -e should kill the wrapper before that.
    assert "Computing per-day RCE trajectory summary" not in res.stdout, (
        "summarizer ran despite mpirun failure — set -e gate is broken."
    )


def test_wrapper_summarizer_fail_default_propagates(tmp_path):
    """iter-100 Codex MEDIUM#1: mpirun OK + summarizer fail +
    ALLOW_SUMMARY_FAILURE=0 (default) → wrapper exits nonzero."""
    res = _run_wrapper_with_stubs(
        tmp_path, mpirun_exit=0, summarizer_exit=7,
        allow_summary_failure="0",
    )
    assert res.returncode == 7, (
        f"summarizer failure was masked: returncode={res.returncode} "
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert "ERROR" in res.stderr


def test_wrapper_summarizer_fail_allow_downgrade(tmp_path):
    """ALLOW_SUMMARY_FAILURE=1 → wrapper exits 0 even when the
    summarizer fails. The escape hatch is intentional for runs
    aborted before any snapshot landed."""
    res = _run_wrapper_with_stubs(
        tmp_path, mpirun_exit=0, summarizer_exit=7,
        allow_summary_failure="1",
    )
    assert res.returncode == 0, (
        f"ALLOW_SUMMARY_FAILURE=1 did not downgrade: "
        f"returncode={res.returncode} stderr={res.stderr!r}"
    )
    assert "WARN" in res.stderr


def test_wrapper_evaluate_dod_threads_into_summarizer(wrapper_text):
    """iter-103 / iter-113 / iter-128: ``EVALUATE_DOD`` must thread
    the right flag into the summarizer invocation. iter-128
    widened to a four-way case:

      EVALUATE_DOD=0         → no flag (default)
      EVALUATE_DOD=1         → ``--evaluate`` (spinup gate)
      EVALUATE_DOD=stability → ``--evaluate --no-plateau-check``
      EVALUATE_DOD=final     → ``--final-dod`` (30-day production)

    Lock the bash ``case`` so a silent revert is caught by the
    < 1 s text test before any nightly hits it."""
    code = _strip_bash_comments(wrapper_text)
    # 1. EVAL_FLAG="" default (else EVAL_FLAG is unset under any
    #    EVALUATE_DOD other than the recognised values, which
    #    set -u would abort).
    assert re.search(r'^\s*EVAL_FLAG\s*=\s*""\s*$', code, re.MULTILINE), (
        "run_rce_30day.sh missing the ``EVAL_FLAG=\"\"`` default."
    )
    # 2. The case statement maps each EVALUATE_DOD value to the
    #    right CLI flag. Lock 1/stability/final.
    assert re.search(
        r'case\s+"\$EVALUATE_DOD"\s+in[\s\S]*?'
        r'1\)\s*EVAL_FLAG\s*=\s*"--evaluate"[\s\S]*?'
        r'stability\)\s*EVAL_FLAG\s*=\s*"--evaluate --no-plateau-check"[\s\S]*?'
        r'final\)\s*EVAL_FLAG\s*=\s*"--final-dod"',
        code,
    ), (
        "run_rce_30day.sh missing the iter-128 ``case "
        "\"$EVALUATE_DOD\" in 1) ... ; stability) ... ; final) "
        "... esac`` dispatch. Without it, EVALUATE_DOD=stability "
        "cannot reach the iter-127 --no-plateau-check summarizer "
        "flag."
    )
    # 3. EVAL_FLAG is threaded into the summarizer argv.
    assert re.search(r'\$EVAL_FLAG\b', code), (
        "run_rce_30day.sh defines EVAL_FLAG but does not pass it "
        "into the summarizer invocation. EVALUATE_DOD=1/stability/final "
        "would be a silent no-op."
    )


def test_wrapper_evaluate_dod_propagates_dod_fail(tmp_path):
    """iter-103 behavioural smoke: EVALUATE_DOD=1 + a summarizer that
    fails (e.g. on FAIL DOD verdict, iter-104 exit code 3) must
    propagate non-zero unless ALLOW_SUMMARY_FAILURE=1 downgrades it.
    Reuses the iter-101 stub harness.

    iter-104: switched the expected summarizer exit code from 1 to 3
    to match the new distinct-code contract (Codex MEDIUM#7). The
    wrapper still propagates the literal status, so the assertion
    is the literal stub code, not a hardcoded ``1``."""
    res = _run_wrapper_with_stubs(
        tmp_path, mpirun_exit=0, summarizer_exit=3,
        allow_summary_failure="0",
    )
    assert res.returncode == 3, (
        f"DOD FAIL (exit 3) did not propagate: returncode={res.returncode} "
        f"stderr={res.stderr!r}"
    )


def test_wrapper_evaluate_dod_insufficient_propagates(tmp_path):
    """iter-104 Codex MEDIUM#3: --evaluate on a too-short trajectory
    exits with EXIT_DOD_INSUFFICIENT=4. The wrapper must propagate
    that literal code (distinct from PASS=0 and FAIL=3) so automation
    can tell "trajectory too short to gate" apart from "trajectory
    gated and failed"."""
    res = _run_wrapper_with_stubs(
        tmp_path, mpirun_exit=0, summarizer_exit=4,
        allow_summary_failure="0",
    )
    assert res.returncode == 4, (
        f"DOD INSUFFICIENT (exit 4) did not propagate: "
        f"returncode={res.returncode} stderr={res.stderr!r}"
    )


def test_wrapper_evaluate_dod_allow_summary_failure_downgrades(tmp_path):
    """iter-103: ALLOW_SUMMARY_FAILURE=1 still covers any non-zero
    summarizer exit (IO=1, DOD FAIL=3, INSUFFICIENT=4). The wrapper
    exits 0 with WARN on stderr. Intentional escape hatch (a developer
    can opt out of DOD-blocking by setting the env). iter-104 exercises
    the DOD FAIL code 3 (post-Codex MEDIUM#7 split)."""
    res = _run_wrapper_with_stubs(
        tmp_path, mpirun_exit=0, summarizer_exit=3,
        allow_summary_failure="1",
    )
    assert res.returncode == 0
    assert "WARN" in res.stderr


@pytest.mark.parametrize(
    "summarizer_exit",
    # iter-172: the iter-103 docstring promises ALLOW_SUMMARY_FAILURE
    # covers IO error (1), DOD FAIL (3) AND INSUFFICIENT (4). iter-103
    # only tested code 3; the 1 + 4 legs were uncovered. A regression
    # that special-cased the downgrade only for code 3 would silently
    # let an IO error or INSUFFICIENT propagate as a wrapper FAIL even
    # when the caller explicitly opted out of DOD-blocking.
    [1, 4],
)
def test_wrapper_allow_summary_failure_covers_all_summarizer_exits(
    tmp_path, summarizer_exit,
):
    """iter-172: ALLOW_SUMMARY_FAILURE=1 downgrades EVERY non-zero
    summarizer exit code (not just DOD FAIL=3) to wrapper exit 0
    with WARN on stderr. Companion to
    test_wrapper_evaluate_dod_allow_summary_failure_downgrades
    (which exercises code 3) so the iter-103 docstring promise is
    fully verified.
    """
    res = _run_wrapper_with_stubs(
        tmp_path, mpirun_exit=0, summarizer_exit=summarizer_exit,
        allow_summary_failure="1",
    )
    assert res.returncode == 0, (
        f"ALLOW_SUMMARY_FAILURE=1 should downgrade summarizer "
        f"exit {summarizer_exit} to wrapper exit 0; got "
        f"{res.returncode}; stderr={res.stderr!r}"
    )
    assert "WARN" in res.stderr


def test_wrapper_evaluate_dod_final_threads_final_dod_flag(tmp_path):
    """iter-113: EVALUATE_DOD=final must reach the summarizer as
    ``--final-dod``. The stub PYBIN echoes its argv so we can
    verify the flag actually made it through the bash dispatch."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "0"
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = "final"
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    # The stub PYBIN echo lands in trajectory.txt (the wrapper
    # redirects > "$OUTPUT/trajectory.txt" 2>&1). Read it back.
    traj_path = out_dir / "trajectory.txt"
    assert traj_path.exists(), (
        f"wrapper did not invoke summarizer; stdout={res.stdout!r} "
        f"stderr={res.stderr!r}"
    )
    text = traj_path.read_text()
    assert "--final-dod" in text, (
        f"EVALUATE_DOD=final did NOT reach the summarizer; "
        f"trajectory.txt={text!r}"
    )
    assert "--evaluate" not in text, (
        f"EVALUATE_DOD=final accidentally also passed --evaluate "
        f"(should be mutually exclusive); trajectory.txt={text!r}"
    )


def test_wrapper_evaluate_dod_1_threads_evaluate_flag(tmp_path):
    """iter-113 sibling: EVALUATE_DOD=1 still reaches the summarizer
    as ``--evaluate`` (back-compat with iter-103)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "0"
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = "1"
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )
    traj = (out_dir / "trajectory.txt").read_text()
    assert "--evaluate" in traj
    assert "--final-dod" not in traj


def test_wrapper_evaluate_dod_rejects_typo(tmp_path):
    """iter-114 Codex MEDIUM#4: ``EVALUATE_DOD=Final`` (or any other
    typo / unrecognised value) used to silently disable grading.
    The wrapper now refuses with exit 1 + a stderr error listing
    the valid values."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "0"
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = "Final"  # typo: capital F
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )
    assert res.returncode == 1, (
        f"EVALUATE_DOD=Final should exit 1; got {res.returncode}. "
        f"stderr={res.stderr!r}"
    )
    assert "EVALUATE_DOD='Final'" in res.stderr, (
        f"stderr should list the offending value verbatim; "
        f"stderr={res.stderr!r}"
    )
    assert "Valid values: 0" in res.stderr, (
        "stderr should list valid values"
    )


def test_wrapper_prints_final_dod_hint_on_30day_default(tmp_path):
    """iter-124: when DAYS>=30 and EVALUATE_DOD=0 (default), the
    wrapper should print a hint on stdout suggesting --final-dod
    as the manual follow-up. Catches the silent-skip class where
    a production-config run lands a trajectory.csv but never grades
    it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "30"        # production-scale
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = "0"  # the default — DOD not auto-gated
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )
    assert res.returncode == 0
    assert "Hint: this is a >=30-day production run" in res.stdout, (
        f"expected the iter-124 hint on a 30-day + EVALUATE_DOD=0 run;\n"
        f"stdout={res.stdout!r}"
    )
    assert "--final-dod" in res.stdout
    # iter-141: hint must also mention --check-log-max-w (criterion 1
    # gate landed iter-137/138). Pre-iter-141 the hint only covered
    # criterion 2 (plateau / MSE drift), leaving criterion 1
    # (max|w|) silently un-checked on 30-day production runs.
    assert "--check-log-max-w" in res.stdout, (
        f"hint should mention --check-log-max-w so users gate "
        f"BOTH DOD criteria; stdout={res.stdout!r}"
    )
    assert "CHECK_LOG_MAX_W=1" in res.stdout


def test_wrapper_no_hint_on_short_run(tmp_path):
    """iter-124 sibling: short runs (DAYS<30) should NOT trigger the
    final-DOD hint."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "5"
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = "0"
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )
    assert res.returncode == 0
    assert "Hint: this is a >=30-day" not in res.stdout, (
        f"5-day smoke should not trigger the 30-day hint; "
        f"stdout={res.stdout!r}"
    )


def test_wrapper_no_hint_when_evaluate_dod_set(tmp_path):
    """iter-124: EVALUATE_DOD=final already grades the run, so the
    hint is redundant — suppress it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "30"
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = "final"  # already grading
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )
    assert res.returncode == 0
    assert "Hint: this is a >=30-day" not in res.stdout, (
        f"EVALUATE_DOD=final should suppress the hint; "
        f"stdout={res.stdout!r}"
    )


def test_wrapper_emit_trajectory_png_default_off(wrapper_text):
    """iter-125: EMIT_TRAJECTORY_PNG defaults to 0 (PNG step skipped)
    so the wrapper stays minimal-dependency. Locks the bash env
    default + the conditional gate around the plot call.

    iter-126: gate widened to ``= "1" || = "strict"`` so the
    propagate-on-fail mode also runs the plot call."""
    code = _strip_bash_comments(wrapper_text)
    # Default env value 0.
    assert re.search(
        r'EMIT_TRAJECTORY_PNG\s*=\s*"\$\{EMIT_TRAJECTORY_PNG:-0\}"',
        code,
    ), (
        "run_rce_30day.sh missing EMIT_TRAJECTORY_PNG=0 default; "
        "the PNG step would default to ON, adding a matplotlib "
        "dependency for runs that don't need the visual."
    )
    # Conditional gate around the plot call — must accept BOTH
    # ``1`` (best-effort) and ``strict`` (propagate-on-fail).
    assert re.search(
        r'if\s*\[\s*"\$EMIT_TRAJECTORY_PNG"\s*=\s*"1"\s*\]'
        r'\s*\|\|\s*'
        r'\[\s*"\$EMIT_TRAJECTORY_PNG"\s*=\s*"strict"\s*\]'
        r'\s*;\s*then',
        code,
    ), (
        "run_rce_30day.sh missing EMIT_TRAJECTORY_PNG gate accepting "
        "BOTH ``1`` (best-effort) and ``strict`` (propagate-on-fail). "
        "Without the disjunction, EMIT_TRAJECTORY_PNG=strict would "
        "be a silent no-op (iter-126 Codex MEDIUM#2 regression risk)."
    )


def test_wrapper_emit_trajectory_png_invokes_plot_rce_log(wrapper_text):
    """iter-125: when the EMIT_TRAJECTORY_PNG=1 branch fires, it
    must invoke scripts/plot_rce_log.py against $OUTPUT."""
    code = _strip_bash_comments(wrapper_text)
    assert re.search(
        r'"\$PYBIN"\s+"\$REPO_ROOT/scripts/plot_rce_log\.py"\s+"\$OUTPUT"',
        code,
    ), (
        "run_rce_30day.sh EMIT_TRAJECTORY_PNG=1 branch must call "
        "``\"$PYBIN\" \"$REPO_ROOT/scripts/plot_rce_log.py\" "
        "\"$OUTPUT\"``."
    )


def test_wrapper_emit_trajectory_png_best_effort_and_strict(wrapper_text):
    """iter-125 + iter-126: PNG render block has two failure modes:

    * ``=1`` best-effort — WARN on fail, no exit-code change.
    * ``=strict`` — ERROR + ``exit "$plot_status"`` to propagate.

    Lock BOTH branches so a future refactor that drops either gets
    caught."""
    code = _strip_bash_comments(wrapper_text)
    assert re.search(r'plot_status\s*=\s*\$\?', code), (
        "run_rce_30day.sh missing plot_status capture."
    )
    # The plot block uses set +e / set -e then branches on
    # ``elif [ "$EMIT_TRAJECTORY_PNG" = "strict" ]; then exit "$plot_status"``.
    assert re.search(
        r'elif\s*\[\s*"\$EMIT_TRAJECTORY_PNG"\s*=\s*"strict"\s*\]'
        r'[\s\S]{0,200}?exit\s+"\$plot_status"',
        code,
    ), (
        "iter-126 MEDIUM#2: EMIT_TRAJECTORY_PNG=strict must "
        "propagate the plot exit code via ``exit \"$plot_status\"``."
    )
    # And the best-effort (=1, fall-through) branch must still WARN.
    assert "WARN: plot_rce_log.py failed" in code, (
        "Best-effort branch missing the WARN on-fail message."
    )


def _run_wrapper_quick(tmp_path, *, days, evaluate_dod):
    """Shared helper for iter-126 hint edge-case tests."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = days
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = evaluate_dod
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )


def test_wrapper_hint_fires_on_30day_spinup_gate(tmp_path):
    """iter-126 Codex MEDIUM#1 fix: EVALUATE_DOD=1 (spinup 5 % gate)
    on a 30-day run is NOT the production 1 % gate. The hint must
    fire so the user knows to also run --final-dod."""
    res = _run_wrapper_quick(tmp_path, days="30", evaluate_dod="1")
    assert res.returncode == 0
    assert "Hint: this is a >=30-day" in res.stdout, (
        f"EVALUATE_DOD=1 on a 30-day run should still trigger the "
        f"final-DOD hint (spinup gate != production gate); "
        f"stdout={res.stdout!r}"
    )


def test_wrapper_hint_fires_on_30day_stability_gate(tmp_path):
    """iter-156: EVALUATE_DOD=stability (spinup gate without plateau
    check) on a 30-day run is also not the production 1 % final-DOD
    gate, so the hint must fire. Companion to the spinup-gate test
    above — covers the third grading mode the wrapper accepts."""
    res = _run_wrapper_quick(tmp_path, days="30", evaluate_dod="stability")
    assert res.returncode == 0
    assert "Hint: this is a >=30-day" in res.stdout, (
        f"EVALUATE_DOD=stability on a 30-day run should still "
        f"trigger the final-DOD hint (stability gate != production "
        f"gate); stdout={res.stdout!r}"
    )
    # iter-156: hint message echoes the user's EVALUATE_DOD value so
    # the failure mode "I set EVALUATE_DOD=stability but the hint
    # still appears" is self-explanatory.
    assert "EVALUATE_DOD=stability" in res.stdout, (
        f"hint should echo the user's EVALUATE_DOD value verbatim; "
        f"stdout={res.stdout!r}"
    )


def test_wrapper_hint_does_not_fire_on_decimal_days_below_30(tmp_path):
    """iter-126 LOW#1: DAYS=29.99 (floors to 29) must NOT trigger
    the hint."""
    res = _run_wrapper_quick(tmp_path, days="29.99", evaluate_dod="0")
    assert res.returncode == 0
    assert "Hint: this is a >=30-day" not in res.stdout


def test_wrapper_hint_fires_on_30dot5_days(tmp_path):
    """iter-126 LOW#1: DAYS=30.5 (floors to 30) MUST trigger the
    hint."""
    res = _run_wrapper_quick(tmp_path, days="30.5", evaluate_dod="0")
    assert res.returncode == 0
    assert "Hint: this is a >=30-day" in res.stdout


def test_wrapper_hint_skipped_on_non_numeric_days(tmp_path):
    """iter-126 LOW#1: malformed DAYS value (non-numeric) must NOT
    crash the wrapper; the hint silently skips via 2>/dev/null on
    the bash arithmetic. mpirun would already have rejected
    --days "abc" upstream, so this is purely defensive."""
    res = _run_wrapper_quick(tmp_path, days="abc", evaluate_dod="0")
    # The stub mpirun ignores its argv so we won't actually crash on
    # the upstream rejection. The hint should just not fire.
    assert res.returncode == 0
    assert "Hint: this is a >=30-day" not in res.stdout


def test_wrapper_emit_trajectory_png_strict_propagates(tmp_path):
    """iter-126 Codex MEDIUM#2 fix: EMIT_TRAJECTORY_PNG=strict
    propagates a PNG-render failure as the wrapper's exit code,
    distinct from the default best-effort ``=1`` mode."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    # PYBIN stub: succeed on the summarizer call (first .py argv),
    # FAIL with exit 42 on the plot call. Use a simple counter
    # approach: the stub script counts invocations via a sibling
    # file.
    pybin = bin_dir / "fake_pybin"
    pybin.write_text(
        "#!/usr/bin/env bash\n"
        f"count_file={tmp_path}/pybin_count\n"
        'if [ ! -f "$count_file" ]; then echo 0 > "$count_file"; fi\n'
        'count=$(cat "$count_file")\n'
        'next=$((count + 1))\n'
        'echo "$next" > "$count_file"\n'
        'echo "STUB:fake_pybin:$@"\n'
        'if [ "$count" -eq 0 ]; then exit 0; else exit 42; fi\n'
    )
    pybin.chmod(0o755)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "0"
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = "0"
    env["EMIT_TRAJECTORY_PNG"] = "strict"
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )
    assert res.returncode == 42, (
        f"EMIT_TRAJECTORY_PNG=strict should propagate exit 42; "
        f"got {res.returncode}; stderr={res.stderr!r}"
    )
    assert "ERROR: plot_rce_log.py failed" in res.stderr


def test_wrapper_evaluate_dod_stability_threads_no_plateau_check(tmp_path):
    """iter-128 / iter-129: EVALUATE_DOD=stability must reach the
    summarizer as ``--evaluate --no-plateau-check`` (both flags,
    not just one). The stub PYBIN echoes its argv so we can verify."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "0"
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = "stability"
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )
    traj = (out_dir / "trajectory.txt").read_text()
    assert "--evaluate" in traj, (
        f"EVALUATE_DOD=stability did NOT reach --evaluate; "
        f"trajectory.txt={traj!r}"
    )
    assert "--no-plateau-check" in traj, (
        f"EVALUATE_DOD=stability did NOT reach --no-plateau-check; "
        f"trajectory.txt={traj!r}"
    )
    assert "--final-dod" not in traj


def test_wrapper_mpirun_fail_propagates_even_with_allow_summary_failure(tmp_path):
    """iter-135: ``ALLOW_SUMMARY_FAILURE=1`` only downgrades a
    summarizer failure; mpirun failures MUST still propagate
    because set -e + pipefail kills the script BEFORE the post-run
    block runs. Without this test a future refactor that adds a
    wrap-around-mpirun ``set +e`` could silently mask mpirun
    failures."""
    res = _run_wrapper_with_stubs(
        tmp_path, mpirun_exit=42, summarizer_exit=0,
        allow_summary_failure="1",
    )
    assert res.returncode != 0, (
        f"mpirun failure was masked under ALLOW_SUMMARY_FAILURE=1: "
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    # mpirun's stub returned 42; bash + pipefail propagates that
    # literal code through ``mpirun ... | tee``. The exact code
    # depends on pipe semantics; just assert non-zero.
    assert "Computing per-day RCE trajectory summary" not in res.stdout, (
        "summarizer ran despite mpirun failure; ALLOW_SUMMARY_FAILURE=1 "
        "must NOT bypass the mpirun guard."
    )


def test_wrapper_help_flag_prints_usage_and_exits_zero(tmp_path):
    """iter-136: --help and -h must print the docstring header
    (line 2..72) without the leading ``# `` marker and exit 0.
    Pre-iter-136 ``--help`` was interpreted as ``OUTPUT=--help``
    and the wrapper would write garbage to that literal path."""
    for flag in ("--help", "-h"):
        res = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
             flag],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
            check=False, timeout=10,
        )
        assert res.returncode == 0, (
            f"--help should exit 0; got {res.returncode} for {flag!r}"
        )
        # The header text should appear without the leading ``# `` marker.
        assert "30-day RCE @ 132x132" in res.stdout, (
            f"{flag} stdout missing docstring header: {res.stdout!r}"
        )
        # Verify the docstring leading ``# `` markers are stripped
        # (output should have a clean first line, no ``# `` prefix).
        first_content_line = next(
            (ln for ln in res.stdout.splitlines() if ln.strip()), "",
        )
        assert not first_content_line.startswith("# "), (
            f"{flag} did not strip leading ``# ``: "
            f"first_line={first_content_line!r}"
        )


def test_wrapper_check_log_max_w_threads_flag(wrapper_text):
    """iter-138: ``CHECK_LOG_MAX_W=1`` must thread
    ``--check-log-max-w`` into the summarizer invocation.

    Lock the bash conditional (default 0 + ``if [ ... = "1" ];
    then EVAL_FLAG="$EVAL_FLAG --check-log-max-w"; fi``)."""
    code = _strip_bash_comments(wrapper_text)
    assert re.search(
        r'CHECK_LOG_MAX_W\s*=\s*"\$\{CHECK_LOG_MAX_W:-0\}"',
        code,
    ), (
        "run_rce_30day.sh missing CHECK_LOG_MAX_W=0 default; "
        "without it the env var would silently default OFF or "
        "trigger set -u abort."
    )
    assert re.search(
        r'if\s*\[\s*"\$CHECK_LOG_MAX_W"\s*=\s*"1"\s*\]\s*;\s*then\s*\n\s*'
        r'EVAL_FLAG\s*=\s*"\$EVAL_FLAG --check-log-max-w"',
        code,
    ), (
        "run_rce_30day.sh missing the iter-138 ``if [ "
        "\"$CHECK_LOG_MAX_W\" = \"1\" ]; then EVAL_FLAG=... fi`` "
        "block. CHECK_LOG_MAX_W=1 would be a silent no-op."
    )


def test_wrapper_check_log_max_w_subprocess_threads_flag(tmp_path):
    """iter-138 behavioural: CHECK_LOG_MAX_W=1 reaches the
    summarizer argv via the stub PYBIN echo."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "0"
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = "0"
    env["CHECK_LOG_MAX_W"] = "1"
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )
    traj = (out_dir / "trajectory.txt").read_text()
    assert "--check-log-max-w" in traj, (
        f"CHECK_LOG_MAX_W=1 did NOT reach the summarizer; "
        f"trajectory.txt={traj!r}"
    )


def test_wrapper_evaluate_dod_final_plus_check_log_max_w_threads_both(tmp_path):
    """iter-155: the wrapper hint message recommends running with
    ``EVALUATE_DOD=final CHECK_LOG_MAX_W=1`` together. Each leg is
    tested independently elsewhere; this asserts the COMBINED
    case threads ``--final-dod --check-log-max-w`` correctly. A
    bash regression like ``EVAL_FLAG="--check-log-max-w"``
    (overwriting instead of appending) would only surface here.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "0"
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = "final"
    env["CHECK_LOG_MAX_W"] = "1"
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )
    traj = (out_dir / "trajectory.txt").read_text()
    assert "--final-dod" in traj, (
        f"EVALUATE_DOD=final leg lost; trajectory.txt={traj!r}"
    )
    assert "--check-log-max-w" in traj, (
        f"CHECK_LOG_MAX_W=1 leg lost; trajectory.txt={traj!r}"
    )


def test_wrapper_emit_trajectory_png_rejects_typo(tmp_path):
    """iter-177: EMIT_TRAJECTORY_PNG=Strict (typo, capital S) used
    to silently fall through to the implicit-skip branch — a user
    expecting strict propagation got a silent no-PNG. The wrapper
    now refuses with exit 1 + stderr error listing valid values
    (mirroring the iter-114 EVALUATE_DOD typo rejection).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "0"
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EMIT_TRAJECTORY_PNG"] = "Strict"  # typo: capital S
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )
    assert res.returncode == 1, (
        f"EMIT_TRAJECTORY_PNG=Strict should exit 1; got "
        f"{res.returncode}. stderr={res.stderr!r}"
    )
    assert "EMIT_TRAJECTORY_PNG='Strict'" in res.stderr, (
        f"stderr should list the offending value verbatim; "
        f"stderr={res.stderr!r}"
    )
    assert "Valid values: 0" in res.stderr, (
        "stderr should list valid values"
    )


def test_wrapper_check_log_max_w_default_off(tmp_path):
    """iter-138: CHECK_LOG_MAX_W defaults to 0 → no flag passed."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir / "mpirun", exit_code=0)
    pybin = bin_dir / "fake_pybin"
    _make_stub(pybin, exit_code=0)
    out_dir = tmp_path / "wrapper_out"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYBIN"] = str(pybin)
    env["DAYS"] = "0"
    env["NX"] = "4"
    env["NY"] = "4"
    env["RANKS"] = "1"
    env["NO_MASS_FIXER"] = "1"
    env["ALLOW_SUMMARY_FAILURE"] = "0"
    env["EVALUATE_DOD"] = "0"
    # CHECK_LOG_MAX_W unset → falls back to 0.
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         str(out_dir)],
        env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=30,
    )
    traj = (out_dir / "trajectory.txt").read_text()
    assert "--check-log-max-w" not in traj, (
        f"--check-log-max-w should NOT be passed when "
        f"CHECK_LOG_MAX_W unset/0; trajectory.txt={traj!r}"
    )


def test_wrapper_help_lists_all_env_vars(tmp_path):
    """iter-143: --help output must include every env var the
    wrapper documents (the iter-136 extracted range capped at
    line 72, dropping the iter-138 + iter-125 + PYBIN docs).
    A future env var addition that doesn't widen the range would
    be caught here."""
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_rce_30day.sh"),
         "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=10,
    )
    assert res.returncode == 0
    # iter-144 / iter-170: every CONSUMED env var (all 17
    # ``X="${X:-VAL}"`` assignments in run_rce_30day.sh) must surface
    # in the --help output. Add new env vars to BOTH the wrapper
    # docstring AND the sed range AND this list in lockstep.
    # iter-170: ``NY`` was missing from this list (the count comment
    # claimed 17, list held 16) — the wrapper docstring's joint
    # ``NX,NY  grid dims`` line satisfied substring ``NX`` only.
    for env_var in (
        "DAYS", "RANKS", "DT", "NX", "NY", "N_ACOUSTIC", "ADVECTION",
        "HYPERDIFF", "BUBBLE_K", "QV_NOISE", "USE_DD",
        "NO_MASS_FIXER", "EVALUATE_DOD", "ALLOW_SUMMARY_FAILURE",
        "EMIT_TRAJECTORY_PNG", "CHECK_LOG_MAX_W", "PYBIN",
    ):
        assert env_var in res.stdout, (
            f"--help missing env var {env_var!r}; the iter-136 sed "
            f"range may need widening to cover newer docs. "
            f"stdout tail: {res.stdout[-500:]!r}"
        )
