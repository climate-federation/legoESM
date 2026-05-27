"""Regression test for the plane CRM 30-day wrapper script defaults.

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


def test_dt_default_matches_iter14_production(wrapper_text):
    """iter-14 measured the 132x132 plane CRM 1-sim-hour PASS at
    dt=5.0 s (725 steps, max|w|=6.1e-3 m/s). The wrapper must
    default to that value; iter-58 found it stale at 1.0 s."""
    dt = _parse_env_default(wrapper_text, "DT")
    assert dt == "5.0", (
        f"run_rce_30day.sh DT default = {dt!r}, expected '5.0' "
        f"(iter-14 + iter-38 production contract)."
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
        "ADVECTION": "upwind1",
        "HYPERDIFF": "5.0e6",
        "BUBBLE_K": "0.0",   # F7 / F10 contract: clean Wing IC
        "QV_NOISE": "0.0",   # F7 / F10 contract: no qv noise
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
    "--acoustic-off-centering": "0.1",  # iter-14/38 production beta
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
