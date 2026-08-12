"""Exit-status regression for the MPAS schedule-cost scan launcher.

The failure this guards is not hypothetical: the first version of the script
captured arm 3's status into ``RC10`` and then ended on a successful ``echo``,
so a scan that produced NO valid s10 result exited 0 and SLURM filed it as
COMPLETED (codex round 2, BLOCKER).  "Tool status is not evidence" cuts both
ways — a launcher that cannot report failure makes every downstream reading
of ``sacct`` a lie.

HOW THIS IS DRIVEN, and why not the obvious way: the bench path in the
launcher is deliberately NOT overridable from the environment.  Making it
overridable (the first attempt) handed a stray exported variable the power to
redirect a real scan to something that exits 0 — reintroducing the very
failure class under test (codex round 4).  Instead this substitutes the
INTERPRETER via ``LEGOESM_PYTHON``, which is an existing production knob that
``_env.sh`` already reads, so the launcher itself carries no test-only seam.

The stub interpreter answers ``_env.sh``'s jax probe, then exits with a
scripted status per arm, recording each arm's full argv so a test can assert
both WHICH arms ran and that each carried its required flags.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SBATCH = (_REPO / "scripts" / "cluster" / "scaling_levante"
           / "mpas_schedule_cost_scan.sbatch")

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="needs bash to run the launcher")


def _run(tmp_path, codes, write_artifact=True):
    """Run the launcher with a stub interpreter that exits ``codes`` per arm.

    ``codes`` is one exit status per bench invocation, in order (arm 1, arm
    2, arm 3).  When ``write_artifact`` the stub also writes a minimal JSON
    to the arm's ``--out`` containing an ``n_ranks`` key, which is what the
    launcher's post-arm artifact check looks for; setting it False simulates
    an interpreter that exits 0 having measured nothing.

    Returns ``(proc, arms)`` where ``arms`` is the recorded argv of each
    bench call.
    """
    log = tmp_path / "calls.txt"
    stub = tmp_path / "stub_python"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        f"codes = {list(codes)!r}\n"
        f"log = {str(log)!r}\n"
        f"write_artifact = {bool(write_artifact)!r}\n"
        "argv = sys.argv[1:]\n"
        # `-c` must behave like a REAL interpreter, not a rubber stamp: both
        # _env.sh's jax probe and the launcher's JSON artifact check go
        # through it, and stubbing -c to exit 0 would make that check pass
        # vacuously in every test here.
        "if argv and argv[0] == '-c':\n"
        "    src = argv[1]\n"
        "    if 'import jax' in src:\n"
        "        sys.exit(0)\n"
        "    sys.argv = ['-c'] + argv[2:]\n"
        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
        "    sys.exit(0)\n"
        "with open(log, 'a') as f:\n"
        "    f.write(json.dumps(argv) + '\\n')\n"
        "n = sum(1 for _ in open(log))\n"
        "rc = codes[n - 1] if n <= len(codes) else 0\n"
        "if write_artifact and rc == 0:\n"
        "    out = argv[argv.index('--out') + 1]\n"
        "    with open(out, 'w') as f:\n"
        "        json.dump({'rows': [{'n_ranks': 64,\n"
        "                       'schedule': {'n_rounds': 12,\n"
        "                                    'max_degree': 12}}]}, f)\n"
        "sys.exit(rc)\n"
    )
    stub.chmod(0o755)

    # Point REPO at a throwaway tree that only SYMLINKS the real scripts.
    # The launcher cd's to $REPO and writes results/a1/mpas_schedule_cost
    # RELATIVE to it, so running these tests against the real repo would
    # create — and, in the stale-artifact test, DELETE — files in the same
    # directory a live scan writes its receipts to (codex round 6).
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir(exist_ok=True)
    link = fake_repo / "scripts"
    if not link.exists():
        link.symlink_to(_REPO / "scripts")

    env = dict(os.environ)
    env["SLURM_SUBMIT_DIR"] = str(_REPO)   # _env.sh is sourced from the real tree
    env["LEGOESM_REPO"] = str(fake_repo)
    env["LEGOESM_PYTHON"] = str(stub)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    proc = subprocess.run(
        ["bash", str(_SBATCH)], env=env, cwd=str(fake_repo),
        capture_output=True, text=True, timeout=600)
    out_dir = fake_repo / "results" / "a1" / "mpas_schedule_cost"

    arms = []
    if log.exists():
        import json
        arms = [json.loads(line) for line in log.read_text().splitlines()
                if line.strip()]
    return proc, arms, out_dir


def _levels(arms):
    return [a[a.index("--subdivision") + 1] for a in arms]


def test_launcher_is_not_redirectable_from_the_environment():
    """The bench path must be hardcoded.

    If it were env-overridable, a stray exported variable could point every
    arm at something that exits 0 and the job would report SCAN_DONE with no
    results — the exact failure this module guards.
    """
    text = _SBATCH.read_text()
    assert "BENCH=scripts/bench/bench_voronoi_partition_methods.py" in text
    assert "LEGOESM_SCHEDULE_COST_BENCH" not in text


@pytest.mark.parametrize("codes, failing_arm", [([1, 0, 0], 1), ([0, 1, 0], 2)])
def test_validation_failure_aborts_before_the_unknown_arm(
        tmp_path, codes, failing_arm):
    """EITHER validation arm failing must abort non-zero and skip arm 3.

    An instrument that misses the known census cannot be trusted on the
    unknown one, so producing an s10 number anyway is worse than none.
    Both arms are exercised: guarding only arm 1 leaves arm 2 unchecked.
    """
    proc, arms, _out = _run(tmp_path, codes)
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert "SCAN_ABORTED_VALIDATION" in proc.stdout
    assert "SCAN_DONE" not in proc.stdout
    assert _levels(arms) == ["8", "9"], (
        f"arm 3 must not run after validation arm {failing_arm} failed: "
        f"{_levels(arms)}")


def test_arm3_failure_is_not_reported_as_success(tmp_path):
    """The original BLOCKER: arm 3 fails, the job must NOT exit 0, and must
    propagate the exact status."""
    proc, arms, _out = _run(tmp_path, [0, 0, 3])
    assert proc.returncode == 3, (
        f"arm-3 status not propagated (got {proc.returncode})\n"
        f"{proc.stdout[-2000:]}")
    assert "SCAN_FAILED_ARM3" in proc.stdout
    assert "SCAN_DONE" not in proc.stdout
    assert _levels(arms) == ["8", "9", "10"], _levels(arms)


def test_full_success_exits_zero_and_runs_every_arm(tmp_path):
    proc, arms, _out = _run(tmp_path, [0, 0, 0])
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert "SCAN_DONE" in proc.stdout
    assert _levels(arms) == ["8", "9", "10"], _levels(arms)


def test_each_arm_carries_the_flags_its_claim_depends_on(tmp_path):
    """The exit plumbing being right is worthless if an arm silently stops
    scoring or stops gating.

    Without this, dropping ``--schedule-cost`` (nothing is scored) or
    ``--expect-rounds`` (the validation arms assert nothing) would leave
    every other test in this module green.
    """
    _, arms, _out = _run(tmp_path, [0, 0, 0])
    assert len(arms) == 3, arms
    for argv in arms:
        assert "--schedule-cost" in argv, argv
        assert "--lloyd" in argv and argv[argv.index("--lloyd") + 1] == "0"
    # Validation arms must actually gate; the unknown arm must not pretend to.
    for argv in arms[:2]:
        assert "--expect-rounds" in argv, argv
        assert argv[argv.index("--expect-rounds") + 1].strip(), argv
    assert "--expect-rounds" not in arms[2], arms[2]
    # The WORKING POINTS are part of the claim: dropping 128, or a method,
    # would leave every other launcher test green and only surface as a
    # failed gate in a real (hours-long) run.
    assert [a[a.index("--rank-counts") + 1] for a in arms] == [
        "64,128", "64,128", "128"]
    for argv in arms:
        assert argv[argv.index("--methods") + 1] == "geometric,sfc,metis", argv


def test_arm_that_exits_zero_without_producing_results_is_caught(tmp_path):
    """An exit code is not evidence an arm measured anything.

    If the interpreter succeeds but writes no JSON (or one with no scored
    rows), the launcher must NOT report SCAN_DONE — otherwise a misconfigured
    run files as COMPLETED with nothing in it, the exact class this module
    exists to prevent.
    """
    proc, arms, _out = _run(tmp_path, [0, 0, 0], write_artifact=False)
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert "no scored rows" in proc.stdout
    assert "SCAN_DONE" not in proc.stdout
    # Both validation arms still RUN (they are independent measurements and
    # running both reports more before aborting); what matters is that the
    # unknown arm 3 is skipped.
    assert _levels(arms) == ["8", "9"], _levels(arms)


def test_stale_artifact_from_a_previous_run_is_removed_before_each_arm(
        tmp_path):
    """A failed rerun must not leave the previous run's JSON in place: a
    stale receipt read as current is worse than a missing one.

    Runs entirely inside the throwaway repo tree (see ``_run``), so it can
    never disturb the directory a live scan writes real receipts to.
    """
    out_dir = tmp_path / "repo" / "results" / "a1" / "mpas_schedule_cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "schedule_cost_s8.json"
    stale.write_text('{"rows": [{"n_ranks": 999, "STALE": true}]}')

    # Arm 1 fails and writes nothing; the stale file must be GONE, not left
    # behind looking like this run's result.
    proc, _, _out = _run(tmp_path, [1, 0, 0], write_artifact=False)
    assert proc.returncode != 0
    assert not stale.exists(), (
        "stale s8 JSON survived a failed arm and would read as current")


def test_artifact_check_parses_json_and_rejects_an_empty_rows_list(tmp_path):
    """The guard must PARSE, not grep.

    A substring check for "n_ranks" passes on {"rows": [], "note":
    "n_ranks"} — an empty result that mentions the key — which is exactly
    the empty-but-successful artifact the guard exists to catch (codex
    round 6).
    """
    log = tmp_path / "calls.txt"
    stub = tmp_path / "stub_python"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        f"log = {str(log)!r}\n"
        "argv = sys.argv[1:]\n"
        "if argv and argv[0] == '-c':\n"
        "    src = argv[1]\n"
        "    if 'import jax' in src:\n"
        "        sys.exit(0)\n"
        "    sys.argv = ['-c'] + argv[2:]\n"
        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
        "    sys.exit(0)\n"
        "with open(log, 'a') as f:\n"
        "    f.write(json.dumps(argv) + '\\n')\n"
        # Exit 0 having written a decoy: valid JSON, mentions the key, but
        # carries no scored row.
        "out = argv[argv.index('--out') + 1]\n"
        "with open(out, 'w') as f:\n"
        "    json.dump({'rows': [], 'note': 'n_ranks'}, f)\n"
        "sys.exit(0)\n"
    )
    stub.chmod(0o755)
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir(exist_ok=True)
    if not (fake_repo / "scripts").exists():
        (fake_repo / "scripts").symlink_to(_REPO / "scripts")

    env = dict(os.environ)
    env["SLURM_SUBMIT_DIR"] = str(_REPO)
    env["LEGOESM_REPO"] = str(fake_repo)
    env["LEGOESM_PYTHON"] = str(stub)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    proc = subprocess.run(["bash", str(_SBATCH)], env=env, cwd=str(fake_repo),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert "no scored rows" in proc.stdout
    assert "SCAN_DONE" not in proc.stdout


def test_launcher_expectations_are_the_full_six_row_census():
    """The gate's strength is its CARDINALITY and content, not its presence.

    Asserting only that --expect-rounds is non-empty (the previous test)
    would still pass if five of the six rows were deleted, because the stub
    never runs the real parser.  Parse the launcher's own literals with the
    production parser and check every method x rank pair is present.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bench_vor_part_gate",
        _REPO / "scripts" / "bench" / "bench_voronoi_partition_methods.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    # The scorer's own docstring census, transcribed once here. Asserting
    # only the KEY SET would let sfc:64=12 rot into sfc:64=999 unnoticed
    # (codex round 6) — the values are the whole point of the gate.
    expected = {
        "S8_EXPECT": {("sfc", 64): 12, ("sfc", 128): 14,
                      ("metis", 64): 13, ("metis", 128): 19,
                      ("geometric", 64): 16, ("geometric", 128): 21},
        "S9_EXPECT": {("sfc", 64): 11, ("sfc", 128): 13,
                      ("metis", 64): 14, ("metis", 128): 18,
                      ("geometric", 64): 14, ("geometric", 128): 18},
    }
    text = _SBATCH.read_text()
    for var in ("S8_EXPECT", "S9_EXPECT"):
        line = next(ln for ln in text.splitlines()
                    if ln.startswith(f"{var}="))
        parsed = bench.parse_expect_rounds(line.split("=", 1)[1].strip('"'))
        assert parsed == expected[var], (
            f"{var} does not match the scorer's reference census "
            f"(spmd_schedule_cost docstring): {sorted(parsed.items())}")


@pytest.mark.parametrize("decoy, why", [
    ('{"rows": {"n_ranks": 1}}',
     "rows is a DICT: iterating it yields keys, so a naive `in` test passes"),
    ('{"rows": [{"n_ranks": 64}]}',
     "a quality-only row with no schedule block is not a SCORED row"),
    ('{"rows": [{"n_ranks": 64, "schedule": {}}]}',
     "a schedule block with no n_rounds carries no score"),
    ('not json at all',
     "unparseable output is not a result"),
])
def test_artifact_guard_rejects_every_non_result_shape(tmp_path, decoy, why):
    """Each decoy is valid-looking output that carries NO schedule score.

    The guard must reject all of them; the first two are the shapes that
    slipped past the earlier substring and `in`-based versions (codex
    rounds 6 and 7).
    """
    stub = tmp_path / "stub_python"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        f"decoy = {decoy!r}\n"
        "argv = sys.argv[1:]\n"
        "if argv and argv[0] == '-c':\n"
        "    src = argv[1]\n"
        "    if 'import jax' in src:\n"
        "        sys.exit(0)\n"
        "    sys.argv = ['-c'] + argv[2:]\n"
        "    exec(compile(src, '<stub-c>', 'exec'), {'__name__': '__main__'})\n"
        "    sys.exit(0)\n"
        "open(argv[argv.index('--out') + 1], 'w').write(decoy)\n"
        "sys.exit(0)\n"
    )
    stub.chmod(0o755)
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir(exist_ok=True)
    if not (fake_repo / "scripts").exists():
        (fake_repo / "scripts").symlink_to(_REPO / "scripts")

    env = dict(os.environ)
    env["SLURM_SUBMIT_DIR"] = str(_REPO)
    env["LEGOESM_REPO"] = str(fake_repo)
    env["LEGOESM_PYTHON"] = str(stub)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    proc = subprocess.run(["bash", str(_SBATCH)], env=env, cwd=str(fake_repo),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode != 0, f"{why}\n{proc.stdout[-1500:]}"
    assert "no scored rows" in proc.stdout, why
    assert "SCAN_DONE" not in proc.stdout, why
