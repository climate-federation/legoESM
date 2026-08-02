"""``run_omip_core2`` restart: CLI wiring + the warm-vs-fresh equivalence gate.

The driver can now ``--restart-save`` a resumable checkpoint and ``--restart-from``
it, so a wallclock-limited job chain integrates forward instead of re-paying the
~60-day cold-start spin-up every leg.

The load-bearing test here is :func:`test_two_leg_resume_matches_continuous`:
run N steps continuously, then run N/2 + restart + N/2 and compare.  It drives
the REAL ocean step (prognostic-TKE closure, so ``state.tke`` is a live carry),
the REAL ``step_sea_ice`` (12-field ``DynamicSeaIceState``, two-way coupled to
the ocean through the open-water fraction), and the REAL step-indexed CORE-II
forcing lookup ``run_omip_core2._idx_t`` — so the three things a restart can
silently lose (the TKE carry, the sea-ice pack, the step counter) all matter to
the answer.

NON-VACUITY is proven by mutation, not asserted: the three
``test_dropping_*_breaks_continuity`` cases deliberately corrupt one restored
piece each and assert the comparison FAILS.  If the equivalence test could pass
with a carry dropped, those tests go red.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ice import SeaIceConfig, init_dynamic_ice_state, step_sea_ice
from legoesm.ice.state import DynamicSeaIceState
from legoesm.ocean.restart import load_run_restart, save_run_restart
from legoesm.ocean.state import OceanSurfaceForcing

from scripts.run.run_omip_core2 import (
    _build_arg_parser,
    _build_atm_to_surface_core2,
    _idx_t,
    orca1_zdftke_config,
)


N_LAT, N_LON, NLEV = 6, 8, 4
H_MAX = 1000.0
# dt=7200 s (2 h) with 8 steps spans 16 h, i.e. THREE of the 6-hourly CORE-II
# record bins that run_omip_core2._idx_t maps `step` onto: steps 1..8 give
# floor(step*7200/21600) = 0,0,1,1,1,2,2,2.
# A shorter dt or fewer steps would leave the record index pinned at 0 and the
# "drop the step counter" mutation below would be VACUOUS —
# test_the_carries_under_test_are_actually_live asserts the index really moves.
# Stability: c=sqrt(g*H)=100 m/s on a 45-deg lon spacing gives a barotropic
# Courant number of ~0.17 at this dt, and the vertical mixing is implicit.
DT = 7200.0
N_REC = 8                    # synthetic 6-hourly CORE-II-like records
U_MIN = 1.0                  # [m/s] wind-speed floor for the ice bulk fluxes
N_TOTAL = 8                  # continuous leg length (4 + 4 around the restart)


# ============================================================================
# argparse round-trip
# ============================================================================

def test_cli_round_trip_restart_flags():
    p = _build_arg_parser()
    d = p.parse_args([])
    assert d.restart_save is None and d.restart_from is None
    assert d.restart_every_days == 0.0
    a = p.parse_args(["--restart-save", "/tmp/r.npz",
                      "--restart-from", "/tmp/parent.npz",
                      "--restart-every-days", "5"])
    assert a.restart_save == "/tmp/r.npz"
    assert a.restart_from == "/tmp/parent.npz"
    assert a.restart_every_days == 5.0


def test_cli_restart_flags_are_documented():
    """Both flags carry help text (a bare flag is undiscoverable in --help)."""
    p = _build_arg_parser()
    helps = {a.dest: (a.help or "") for a in p._actions}
    assert "resumable" in helps["restart_save"].lower()
    assert "resume" in helps["restart_from"].lower()
    assert "cadence" in helps["restart_every_days"].lower()


# ============================================================================
# Source-revision provenance (codex r6 MEDIUM)
# ============================================================================

@pytest.fixture
def hermetic_git(tmp_path, monkeypatch):
    """Neutralise the AMBIENT git configuration for this test (codex r7 LOW).

    Without this the fixture repos inherit the developer's ``~/.gitconfig``
    and the system config — a global ``core.hooksPath``, a commit template, a
    ``commit.gpgsign``, or an ``init.defaultBranch`` policy can make
    ``git commit`` fail or behave differently on someone else's machine, and
    the test would then be measuring the environment rather than the helper.
    """
    empty = tmp_path / "_no_git_config"
    empty.mkdir()
    hooks = tmp_path / "_no_hooks"
    hooks.mkdir()
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(empty))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "restart test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "restart test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@example.invalid")
    # Repository-LOCATION and COMMAND-SCOPE config injection (codex r8 LOW):
    # `-c`-equivalent env config (GIT_CONFIG_COUNT/KEY_n/VALUE_n and the older
    # GIT_CONFIG_PARAMETERS) overrides even GIT_CONFIG_GLOBAL, so clearing the
    # file-scope variables alone is not hermetic.
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                "GIT_CEILING_DIRECTORIES", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS",
                "GIT_ATTR_NOSYSTEM", "GIT_NAMESPACE"):
        monkeypatch.delenv(var, raising=False)
    n_cfg = os.environ.get("GIT_CONFIG_COUNT")
    if n_cfg:
        for i in range(int(n_cfg)):
            monkeypatch.delenv(f"GIT_CONFIG_KEY_{i}", raising=False)
            monkeypatch.delenv(f"GIT_CONFIG_VALUE_{i}", raising=False)
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    return str(hooks)


def _init_git_repo(root, hooks_dir, content="x = 1\n"):
    """Create a one-commit git repo at ``root``; return a runner for it."""
    import subprocess

    def g(*a):
        return subprocess.run(["git", "-C", str(root), *a],
                              capture_output=True, text=True, check=True)

    g("init", "-q")
    g("config", "core.hooksPath", hooks_dir)   # no ambient hooks
    g("config", "commit.gpgsign", "false")
    g("config", "user.email", "t@example.invalid")
    g("config", "user.name", "restart test")
    (root / "src.py").write_text(content)
    g("add", "src.py")
    g("commit", "-q", "-m", "init")
    return g


def test_source_revision_reports_a_status_failure_rather_than_a_clean_tree(
        monkeypatch):
    """codex r7 LOW: the ``-dirty-unknown`` branch (HEAD resolved but
    ``git status`` did not) had no test.  A status failure must NOT be reported
    as a clean tree, and a timeout must fall through to 'unavailable'."""
    import subprocess

    from scripts.run.run_omip_core2 import (
        _SOURCE_REV_UNAVAILABLE, _source_revision,
    )

    sha = "c" * 40
    real_run = subprocess.run

    def _fake(cmd, **kw):
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 128, "", "fatal: nope")
        if "--show-toplevel" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "/some/root\n", "")
        return subprocess.CompletedProcess(cmd, 0, sha + "\n", "")

    monkeypatch.setattr(subprocess, "run", _fake)
    assert _source_revision("/anywhere") == f"{sha}-dirty-unknown"

    def _timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 10)

    monkeypatch.setattr(subprocess, "run", _timeout)
    assert _source_revision("/anywhere") == _SOURCE_REV_UNAVAILABLE
    monkeypatch.setattr(subprocess, "run", real_run)


def test_source_revision_flags_a_mixed_driver_and_package_tree(monkeypatch):
    """codex r7 MEDIUM: the driver script and the imported ``legoesm`` packages
    need not come from the same checkout — every sbatch wrapper sets PYTHONPATH
    explicitly, and a wrong value runs this driver against ANOTHER worktree's
    model code with no warning.  One sha cannot describe both."""
    import subprocess

    from scripts.run import run_omip_core2 as _c2

    driver_sha, pkg_sha = "d" * 40, "e" * 40
    here = str(Path(_c2.__file__).resolve().parent)

    def _fake(cmd, **kw):
        cwd = cmd[2]                       # ["git", "-C", <cwd>, ...]
        driver_side = cwd == here
        if "--show-toplevel" in cmd:
            root = "/tree/driver" if driver_side else "/tree/packages"
            return subprocess.CompletedProcess(cmd, 0, root + "\n", "")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(
            cmd, 0, (driver_sha if driver_side else pkg_sha) + "\n", "")

    monkeypatch.setattr(subprocess, "run", _fake)
    got = _c2._source_revision()
    assert got == f"{driver_sha}+mixedtree:{pkg_sha[:12]}", got
    # ...and the ambiguity must be reported, not silently equal.
    assert _c2._revision_ambiguity(got) is not None
    assert _c2._source_revision_drift_note(got, got) is not None

    # CONTROL: same toplevel on both sides -> the plain sha, and silent.
    def _same_root(cmd, **kw):
        if "--show-toplevel" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "/tree/one\n", "")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, driver_sha + "\n", "")

    monkeypatch.setattr(subprocess, "run", _same_root)
    assert _c2._source_revision() == driver_sha
    assert _c2._revision_ambiguity(driver_sha) is None
    assert _c2._source_revision_drift_note(driver_sha, driver_sha) is None


def test_mixed_tree_check_covers_every_legoesm_namespace_portion(monkeypatch):
    """codex r8 MEDIUM: ``legoesm`` is a PEP-420 NAMESPACE package spread over
    ``src/legoesm`` plus every ``packages/*/legoesm``.  Sampling one module's
    directory would miss a PYTHONPATH that pointed, say, legoesm.ice at a
    different worktree while legoesm.ocean matched."""
    import subprocess

    import legoesm

    from scripts.run import run_omip_core2 as _c2

    portions = [str(Path(p).resolve()) for p in legoesm.__path__]
    assert len(portions) > 1, (
        "legoesm resolved to a single path — this test cannot distinguish "
        "'checks all portions' from 'checks one'")
    odd_one = portions[-1]
    driver_sha, odd_sha = "1" * 40, "9" * 40
    here = str(Path(_c2.__file__).resolve().parent)

    probed: list[str] = []

    def _fake(cmd, **kw):
        cwd = cmd[2]
        probed.append(cwd)
        odd = cwd == odd_one
        if "--show-toplevel" in cmd:
            root = "/tree/odd" if odd else "/tree/main"
            return subprocess.CompletedProcess(cmd, 0, root + "\n", "")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(
            cmd, 0, (odd_sha if odd else driver_sha) + "\n", "")

    monkeypatch.setattr(subprocess, "run", _fake)
    got = _c2._source_revision()
    assert got == f"{driver_sha}+mixedtree:{odd_sha[:12]}", got
    # It really did probe the driver AND every namespace portion.
    assert here in probed
    for p in portions:
        assert p in probed, f"namespace portion {p} was never probed"


def test_source_revision_strips_inherited_git_location_env(monkeypatch):
    """codex r8 MEDIUM: ``git -C <dir>`` does NOT override ``GIT_DIR`` /
    ``GIT_WORK_TREE`` / ``GIT_COMMON_DIR``.  Under a hook or wrapper that
    exports them, both probes would describe an unrelated repository and this
    function would return a confidently WRONG plain sha."""
    import subprocess

    from scripts.run.run_omip_core2 import _source_revision

    monkeypatch.setenv("GIT_DIR", "/somewhere/else/.git")
    monkeypatch.setenv("GIT_WORK_TREE", "/somewhere/else")
    monkeypatch.setenv("GIT_COMMON_DIR", "/somewhere/else/.git")
    monkeypatch.setenv("KEEP_ME", "yes")

    seen: dict = {}

    def _fake(cmd, **kw):
        seen.update(kw.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "f" * 40 + "\n", "")

    monkeypatch.setattr(subprocess, "run", _fake)
    _source_revision("/anywhere")
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
        assert var not in seen, f"{var} leaked into the git child environment"
    assert seen.get("KEEP_ME") == "yes", (
        "the whole environment was dropped; only repository-discovery "
        "variables should be stripped")


def test_source_revision_scopes_dirty_state_and_explicit_failure(
        tmp_path, hermetic_git):
    """codex r6 MEDIUM: the old capture was ``git rev-parse HEAD`` with no
    ``-C``, so it resolved against the CWD — a job launched from ``$HOME`` or
    from a different worktree recorded ANOTHER repository's HEAD.  It also
    never detected a dirty working tree (a SHA describes committed content
    only) and failed SILENTLY to ``None``, which is indistinguishable from
    "this archive predates the field" — so both sides skipped the comparison
    and the resume looked checked when nothing had been checked.
    """
    from scripts.run.run_omip_core2 import (
        _SOURCE_REV_UNAVAILABLE, _source_revision,
    )

    a, b = tmp_path / "repo_a", tmp_path / "repo_b"
    a.mkdir()
    b.mkdir()
    ga = _init_git_repo(a, hermetic_git, "x = 1\n")
    _init_git_repo(b, hermetic_git, "x = 2\n")
    head_a = ga("rev-parse", "HEAD").stdout.strip()

    # SCOPING: each call describes the tree it was POINTED AT, independent of
    # the process CWD (the two repos have different HEADs by construction).
    assert _source_revision(a) == head_a
    assert _source_revision(b) != head_a

    # DIRTY: a modified TRACKED file means the SHA no longer describes the
    # code being executed.
    (a / "src.py").write_text("x = 999\n")
    assert _source_revision(a) == f"{head_a}-dirty"

    # ...but an UNTRACKED file is deliberately NOT dirty (git describe --dirty
    # semantics).  This repo always carries hundreds of untracked scratch
    # scripts, so counting them would pin the marker permanently on and make
    # the warning uninformative.
    ga("checkout", "--", "src.py")
    (a / "scratch.sbatch").write_text("#!/bin/bash\n")
    assert _source_revision(a) == head_a

    # EXPLICIT FAILURE: a path that cannot be resolved records a value, never
    # None — so the resume can say the check could not RUN.
    missing = _source_revision(tmp_path / "does_not_exist")
    assert missing == _SOURCE_REV_UNAVAILABLE
    assert missing is not None


def test_source_revision_of_this_checkout_is_recorded_and_scoped():
    """The production call takes no argument and must describe THIS driver's
    checkout — i.e. the tree that supplies ``run_omip_core2.py``."""
    import subprocess

    from scripts.run import run_omip_core2 as _c2

    here = Path(_c2.__file__).resolve().parent
    got = _c2._source_revision()
    probe = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
    if probe.returncode != 0:              # not a checkout (e.g. installed)
        assert got == _c2._SOURCE_REV_UNAVAILABLE
        return
    sha = probe.stdout.strip()
    assert got.startswith(sha)
    # The suffix, when present, must be one this build actually produces
    # (codex r7 LOW: the earlier assertion rejected the valid
    # '-dirty-unknown' fallback and would have gone red on a real status
    # failure).  Anything else means a marker was added without updating the
    # consumers that branch on it.
    suffix = rest = got[len(sha):]
    for marker in ("-dirty-unknown", "-dirty"):
        if rest.startswith(marker):
            rest = rest[len(marker):]
            break
    if rest.startswith("+mixedtree:"):
        rest = ""
    assert rest == "", f"unrecognised source-revision suffix {suffix!r}"
    if suffix:
        assert _c2._revision_ambiguity(got) is not None, (
            f"suffix {suffix!r} is not reported as an ambiguity")


def test_resume_drift_note_distinguishes_unknown_from_a_match():
    """codex r6 MEDIUM: with the revision stored as ``None`` on failure, "the
    check could NOT run" was indistinguishable from "the check ran and
    matched" — both sides were falsy, the comparison was skipped, and the
    resume looked verified.  The three outcomes must stay distinct.

    Behavioural, not source-inspecting: it calls the function ``main`` calls.
    """
    from scripts.run.run_omip_core2 import (
        _SOURCE_REV_UNAVAILABLE, _source_revision_drift_note as note,
    )

    sha, other = "a" * 40, "b" * 40

    # Equal + clean: the only case that may be silent.
    assert note(sha, sha) is None

    # Genuine drift: warns and names BOTH revisions.
    n = note(other, sha)
    assert n is not None and other[:20] in n and sha[:20] in n

    # Either side unknown -> must NOT read as "checked and equal".
    for pair in ((None, sha), ("", sha), (_SOURCE_REV_UNAVAILABLE, sha),
                 (sha, _SOURCE_REV_UNAVAILABLE),
                 (_SOURCE_REV_UNAVAILABLE, _SOURCE_REV_UNAVAILABLE)):
        n = note(*pair)
        assert n is not None, f"{pair} silently skipped the drift check"
        assert "SKIPPED" in n, f"{pair} did not say the check could not run"

    # Equal but AMBIGUOUS: identical markers do not identify identical code,
    # so these may not be silent either.
    for amb in (f"{sha}-dirty", f"{sha}-dirty-unknown",
                f"{sha}+mixedtree:0123456789ab"):
        n = note(amb, amb)
        assert n is not None, f"{amb} was silently accepted as a match"


def test_restart_fingerprint_covers_the_resolved_forcing_archive():
    """codex r8 MEDIUM: with --forcing-path unset the CORE-II loader falls back
    to an environment/home-dependent cache directory, so two legs whose command
    lines are IDENTICAL (both hashing ``forcing_path=None``) could read
    different archives and still produce matching fingerprints.

    main() must fold the RESOLVED path in, and it must get it from the loader's
    own helper — a re-derived copy of the resolution rule would drift.
    """
    import inspect

    from scripts.run import run_omip_core2 as _c2

    src = inspect.getsource(_c2.main)
    assert "core2_nyf_path" in src, (
        "main() no longer resolves the forcing archive for the fingerprint")
    assert "forcing_archive=" in src, (
        "the resolved forcing archive is no longer hashed into the restart "
        "configuration fingerprint")
    # It must be inside the fingerprint block, i.e. before the digest is taken.
    lines = src.splitlines()
    use = next(i for i, ln in enumerate(lines) if "forcing_archive=" in ln)
    digest = next(i for i, ln in enumerate(lines)
                  if "_restart_cfg_fp = " in ln and "None" not in ln)
    assert use < digest, "the forcing archive is appended after the digest"


def test_env_fingerprint_includes_gates_and_excludes_infrastructure():
    """BOTH DIRECTIONS, because each failure mode is bad in a different way.

    Including too LITTLE (codex r9 HIGH): several numerics levers are ENV-gated
    rather than CLI flags, so two legs with identical command lines integrate
    different numerics and the fingerprint still matches.

    Including too MUCH (codex tail-round RED): ~120 ``LEGOESM_*`` variables
    exist and many are per-job infrastructure — cache dirs, ports, CPU counts —
    so a BARE prefix sweep hard-aborts every legitimate chained restart,
    breaking exactly the feature the fingerprint protects.

    Behavioural, on the helper ``main`` calls — not source inspection.
    """
    from scripts.run.run_omip_core2 import _restart_env_items

    env = {
        # numerics gates — MUST be hashed
        "LEGOESM_BAROCLINIC_F32": "1",
        "LEGOESM_VMIX_F32_SOLVE": "1",
        "LEGOESM_VMIX_BATCHED": "1",
        "LEGOESM_TRACER_PAIR": "1",
        "LEGOESM_NO_MASS_FIX": "1",
        # per-job infrastructure — MUST NOT be hashed
        "LEGOESM_JIT_CACHE_DIR": "/scratch/job123",
        "LEGOESM_CACHE_DIR": "/scratch/job123/cache",
        "LEGOESM_MESH_CACHE_DIR": "/scratch/job123/mesh",
        "LEGOESM_DATA_DIR": "/burg/data",
        "LEGOESM_ETOPO_PATH": "/burg/etopo.nc",
        "LEGOESM_COORD_PORT": "45001",
        "LEGOESM_NCPUS": "24",
        "LEGOESM_NGPUS": "2",
        "LEGOESM_SLURM_ACCOUNT": "glab",
        "LEGOESM_PYTHON": "/env/bin/python",
        "LEGOESM_PROFILE_MPI": "1",
        # not ours at all
        "PATH": "/usr/bin",
    }
    got = dict(_restart_env_items(env))
    for k in ("LEGOESM_BAROCLINIC_F32", "LEGOESM_VMIX_F32_SOLVE",
              "LEGOESM_VMIX_BATCHED", "LEGOESM_TRACER_PAIR",
              "LEGOESM_NO_MASS_FIX"):
        assert k in got, f"{k} changes the trajectory and must be fingerprinted"
    for k in ("LEGOESM_JIT_CACHE_DIR", "LEGOESM_CACHE_DIR",
              "LEGOESM_MESH_CACHE_DIR", "LEGOESM_DATA_DIR",
              "LEGOESM_ETOPO_PATH", "LEGOESM_COORD_PORT", "LEGOESM_NCPUS",
              "LEGOESM_NGPUS", "LEGOESM_SLURM_ACCOUNT", "LEGOESM_PYTHON",
              "LEGOESM_PROFILE_MPI", "PATH"):
        assert k not in got, (
            f"{k} differs per job; fingerprinting it false-aborts every "
            "legitimate chained restart")

    # DETERMINISTIC: sorted, so the digest cannot depend on env iteration order.
    assert _restart_env_items(env) == sorted(_restart_env_items(env))

    # FAIL-CLOSED: a lever added later, carrying no infrastructure suffix, is
    # covered without anyone touching this code.
    assert ("LEGOESM_SOME_FUTURE_NUMERICS_LEVER", "1") in _restart_env_items(
        {"LEGOESM_SOME_FUTURE_NUMERICS_LEVER": "1"})


def test_restart_fingerprint_wires_the_env_sweep_before_the_digest():
    """The helper above is only useful if ``main`` actually hashes its result,
    and does so BEFORE taking the digest."""
    import inspect

    from scripts.run import run_omip_core2 as _c2

    src = inspect.getsource(_c2.main)
    assert "_restart_env_items()" in src, (
        "main() no longer folds the LEGOESM_* environment into the restart "
        "configuration fingerprint")
    lines = src.splitlines()
    use = next(i for i, ln in enumerate(lines) if "_restart_env_items()" in ln)
    digest = next(i for i, ln in enumerate(lines)
                  if "_restart_cfg_fp = " in ln and "None" not in ln)
    assert use < digest, "the environment sweep runs after the digest is taken"


def test_mixed_tree_tag_is_unknown_when_the_root_cannot_be_resolved(
        monkeypatch):
    """codex r9 LOW: the docstring promises ``+mixedtree:unknown`` for a
    portion whose repository root does not resolve; it was tagged with that
    portion's SHA instead, which claims more than is known."""
    import subprocess

    from scripts.run import run_omip_core2 as _c2

    here = str(Path(_c2.__file__).resolve().parent)
    driver_sha = "7" * 40

    def _fake(cmd, **kw):
        cwd = cmd[2]
        if "--show-toplevel" in cmd:
            if cwd == here:
                return subprocess.CompletedProcess(cmd, 0, "/tree/main\n", "")
            return subprocess.CompletedProcess(cmd, 128, "", "fatal")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, driver_sha + "\n", "")

    monkeypatch.setattr(subprocess, "run", _fake)
    got = _c2._source_revision()
    assert got == f"{driver_sha}+mixedtree:unknown", got
    assert _c2._revision_ambiguity(got) is not None


def test_provenance_probe_work_and_output_are_bounded(monkeypatch):
    """codex r9 MEDIUM: each foreign portion costs up to three 10 s git
    subprocesses at SETUP, and the recorded string goes into the archive.  The
    common case (one checkout) must cost ONE command per portion, and neither
    the probe count nor the tag list may grow without bound."""
    import subprocess

    from scripts.run import run_omip_core2 as _c2

    here = str(Path(_c2.__file__).resolve().parent)
    calls: list[list[str]] = []

    # (a) FAST PATH: every portion in the driver's tree -> one command each.
    def _same(cmd, **kw):
        calls.append(cmd)
        if "--show-toplevel" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "/tree/one\n", "")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "8" * 40 + "\n", "")

    monkeypatch.setattr(subprocess, "run", _same)
    assert _c2._source_revision() == "8" * 40
    per_portion = [c for c in calls if c[2] != here]
    assert all("--show-toplevel" in c for c in per_portion), (
        "the same-checkout fast path issued more than the root probe: "
        f"{[c[3:] for c in per_portion if '--show-toplevel' not in c]}")

    # (b) BOUNDED OUTPUT: many distinct foreign roots -> a truncated tag list.
    monkeypatch.setattr(_c2, "_MAX_TREE_TAGS", 2)
    counter = {"n": 0}

    def _all_different(cmd, **kw):
        cwd = cmd[2]
        if "--show-toplevel" in cmd:
            if cwd == here:
                return subprocess.CompletedProcess(cmd, 0, "/tree/main\n", "")
            counter["n"] += 1
            return subprocess.CompletedProcess(
                cmd, 0, f"/tree/other{counter['n']}\n", "")
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        sha = ("9" * 39 + str(counter["n"] % 10)) if cwd != here else "8" * 40
        return subprocess.CompletedProcess(cmd, 0, sha + "\n", "")

    monkeypatch.setattr(subprocess, "run", _all_different)
    got = _c2._source_revision()
    tags = got.split("+mixedtree:")[1].split(",")
    # Bounded: at most _MAX_TREE_TAGS real tags, plus ONE explicit marker.
    assert len(tags) <= 3, f"tag list not truncated: {tags}"
    if len(tags) > 2:
        # ...and truncation is ANNOUNCED, not silent — a list that merely
        # stopped early would look complete (codex tail round).
        assert tags[-1].endswith("-more"), (
            f"tags were dropped without saying so: {tags}")
    # CANONICAL: sorted, so the recorded provenance does not depend on
    # legoesm.__path__ order.
    real = [t for t in tags if not t.endswith("-more")]
    assert real == sorted(real), f"tag order is not canonical: {real}"


def test_scan_lane_refuses_restarts():
    """codex r2 HIGH: the --scan-block body steps model._step_impl directly and
    never seeds the scan carry, so it can PROMOTE an optional slot None->Field
    mid-block that the restart neither records nor advances.  Restarts on that
    lane are refused outright; the guard text must name the flag and the fix."""
    import inspect

    from scripts.run import run_omip_core2 as _c2

    src = inspect.getsource(_c2.main)
    assert "--restart-save/--restart-from is not supported with" in src, (
        "the scan-lane restart refusal is gone from main()")
    # And it must sit INSIDE the use_scan branch, before the block builder.
    lines = src.splitlines()
    scan = next(i for i, ln in enumerate(lines) if ln.strip() == "if use_scan:")
    guard = next(i for i, ln in enumerate(lines)
                 if "is not supported with" in ln)
    build = next(i for i, ln in enumerate(lines)
                 if "build_omip2_scan_block_fn(" in ln and "import" not in ln)
    assert scan < guard < build, (
        "the refusal must execute inside the scan branch before the lane runs")


# ============================================================================
# Two-leg warm-vs-fresh equivalence
# ============================================================================

def _setup():
    """Tiny lat-lon C-grid ocean with the PROGNOSTIC TKE closure + dynamic ice."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanConfig, LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z_coord = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)
    # prognostic=True makes state.tke a LIVE carry (Mode-A: one backward-Euler
    # TKE solve per model step, seeded from the carried field) — without it the
    # "drop tke" mutation below would be vacuous.  The closure is read from
    # config.physics.vertical_mixing (what _tke_prognostic_active inspects),
    # with every OTHER physics module off so the test exercises the carry, not
    # a full physics stack.
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        SurfaceForcingConfig,
    )
    vmix = VerticalMixingConfig(scheme="tke",
                                tke=orca1_zdftke_config(prognostic=True))
    phys = OceanPhysicsConfig(
        vertical_mixing=vmix,
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        shortwave_penetration=None,
    )
    cfg = LatLonCGridOceanConfig(implicit_vertical_mixing=True, physics=phys)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=-1.0, T_deep=1.0, S_uniform=34.0,
        H_max=H_MAX,
    )
    # brine ON (as run_omip_core2 --prognostic-sea-ice does) so `step_sea_ice`
    # takes the NEW-PHYSICS path and keeps the 12-field DynamicSeaIceState.
    # With every gate off and dynamics="none" the dispatcher falls to the slab
    # branch, which DOWNGRADES the state to the 3-field SeaIceState — 9 of the
    # 12 fields under test would then not exist.  dynamics/transport stay
    # "none" so no grid-specific strain-rate operators are required.
    from legoesm.ice.config import BrineConfig, SnowConfig
    ice_cfg = SeaIceConfig(dynamics="none", transport="none",
                           brine=BrineConfig(enabled=True),
                           snow=SnowConfig(enabled=True))
    ice0 = init_dynamic_ice_state((N_LAT, N_LON))
    # Seed a real pack: a cold start (h=0, T_ice=260 K, S_ice=0) is exactly the
    # state a dropped ice restart would rebuild, so the reference must NOT be
    # that.  Thickness/concentration vary in latitude (polar-heavy).
    lat_w = np.linspace(1.0, 0.0, N_LAT)[:, None] * np.ones((1, N_LON))
    ice0 = ice0._replace(
        h_ice=ice0.h_ice.replace(data=jnp.asarray(0.8 * lat_w)),
        concentration=ice0.concentration.replace(
            data=jnp.asarray(0.7 * lat_w)),
        T_ice=ice0.T_ice.replace(
            data=jnp.asarray(constants.T_freeze - 8.0 * lat_w)),
        h_snow=ice0.h_snow.replace(data=jnp.asarray(0.1 * lat_w)),
    )
    return grid, z_coord, model, state, ice_cfg, ice0


def _forcing_stack(seed=3):
    """``N_REC`` synthetic CORE-II-like records, indexed per step by _idx_t."""
    rng = np.random.default_rng(seed)
    shape = (N_REC, N_LAT, N_LON)
    return {
        "u10": 6.0 + 3.0 * rng.standard_normal(shape),
        "v10": 2.0 + 3.0 * rng.standard_normal(shape),
        # Records differ strongly in T_air so the step->record mapping is
        # observable: a wrong step counter picks a visibly different record.
        "T_air": constants.T_freeze - 6.0
                 + 8.0 * np.linspace(-1.0, 1.0, N_REC)[:, None, None]
                 * np.ones((1, N_LAT, N_LON)),
        "q_air": 2.0e-3 * np.ones(shape),
        "sw_down": 120.0 + 100.0 * rng.random(shape),
        "lw_down": 250.0 + 20.0 * rng.random(shape),
        "precip": 1.0e-5 * rng.random(shape),
    }


def _leg(model, state, ice_state, ice_cfg, stack, step0, nsteps):
    """Integrate [step0+1, step0+nsteps] exactly as the driver's host loop does.

    Two-way ice<->ocean coupling: the ocean SST drives the ice basal exchange,
    and the ice concentration scales the open-water fraction that the surface
    stress / heat / salt reach the ocean through.  So a dropped ice restart
    perturbs the OCEAN answer too, not just the ice diagnostics.
    """
    for step in range(step0 + 1, step0 + nsteps + 1):
        it = _idx_t(step, DT, N_REC)
        forc = {k: v[it] for k, v in stack.items()}
        atm = _build_atm_to_surface_core2(forc, ramp=1.0)
        sst_K = state.T.data[:, :, 0] + constants.T_freeze
        ocn_u = state.u.data[:, :-1, 0]
        ocn_v = state.v.data[:-1, :, 0]
        ice_state, resp = step_sea_ice(ice_state, atm, sst_K, ocn_u, ocn_v,
                                       ice_cfg, U_MIN, DT)
        # step_sea_ice DOWNGRADES to the 3-field SeaIceState on its slab
        # branch; if that ever happened here the 12-field coverage this test
        # claims would silently shrink to 3.
        assert isinstance(ice_state, DynamicSeaIceState), (
            f"step_sea_ice returned {type(ice_state).__name__}; the test's "
            "SeaIceConfig no longer selects the 12-field dynamic state")
        f_ocn = 1.0 - ice_state.concentration.data
        # SIGN CONVENTION: TileResponse.shflx is positive UP (surface -> atm),
        # OceanSurfaceForcing.q_net is positive INTO the ocean, hence the
        # negation.  Only the open-water fraction reaches the ocean.
        sf = OceanSurfaceForcing(
            sw_down=jnp.asarray(forc["sw_down"]) * f_ocn,
            q_net=-jnp.asarray(resp.shflx) * f_ocn,
            tau_x=jnp.asarray(1.0e-3 * forc["u10"]) * f_ocn,
            tau_y=jnp.asarray(1.0e-3 * forc["v10"]) * f_ocn,
        )
        state = model.step(state, DT, surface_forcing=sf)
    return state, ice_state


def _assert_states_match(got, want, got_ice, want_ice, *, atol=1e-10):
    for name in ("T", "S", "u", "v", "eta"):
        np.testing.assert_allclose(
            np.asarray(getattr(got, name).data),
            np.asarray(getattr(want, name).data), atol=atol, rtol=0,
            err_msg=f"ocean {name} diverged after resume")
    assert (got.tke is None) == (want.tke is None)
    if want.tke is not None:
        np.testing.assert_allclose(np.asarray(got.tke.data),
                                   np.asarray(want.tke.data),
                                   atol=atol, rtol=0,
                                   err_msg="TKE carry diverged after resume")
    for name in DynamicSeaIceState._fields:
        np.testing.assert_allclose(
            np.asarray(getattr(got_ice, name).data),
            np.asarray(getattr(want_ice, name).data), atol=atol, rtol=0,
            err_msg=f"sea ice {name} diverged after resume")


@pytest.fixture(scope="module")
def _legs():
    """(continuous, restart-file inputs) — built once, reused by every case."""
    _, _, model, state0, ice_cfg, ice0 = _setup()
    stack = _forcing_stack()
    cont_state, cont_ice = _leg(model, state0, ice0, ice_cfg, stack, 0, N_TOTAL)
    half_state, half_ice = _leg(model, state0, ice0, ice_cfg, stack, 0,
                                N_TOTAL // 2)
    return dict(model=model, ice_cfg=ice_cfg, stack=stack, template=state0,
                ice_template=ice0, cont_state=cont_state, cont_ice=cont_ice,
                half_state=half_state, half_ice=half_ice)


def test_the_carries_under_test_are_actually_live(_legs):
    """Guard against a vacuous equivalence test: the TKE carry must really be
    populated, the ice must really have evolved, and the forcing must really
    depend on the step index."""
    assert _legs["half_state"].tke is not None, (
        "prognostic TKE is not carrying — the 'drop tke' mutation would be "
        "vacuous")
    assert not np.allclose(np.asarray(_legs["half_ice"].h_ice.data),
                           np.asarray(_legs["ice_template"].h_ice.data)), (
        "the ice pack did not evolve over the first leg")
    # The step->record map must actually move within one leg.
    idx = {_idx_t(s, DT, N_REC) for s in range(1, N_TOTAL + 1)}
    assert len(idx) > 1, "forcing record is constant; the step counter is inert"


def test_two_leg_resume_matches_continuous(tmp_path, _legs):
    """N steps continuous == N/2 + save/load + N/2, ocean AND ice AND carries."""
    path = tmp_path / "restart.npz"
    save_run_restart(path, _legs["half_state"], step=N_TOTAL // 2,
                     time_days=(N_TOTAL // 2) * DT / 86400.0,
                     grid_type="latlon", dt_seconds=DT,
                     n_forcing_records=N_REC, ice_state=_legs["half_ice"])
    warm_state, warm_ice, meta = load_run_restart(
        path, _legs["template"], ice_template=_legs["ice_template"],
        grid_type="latlon", dt_seconds=DT, n_forcing_records=N_REC)
    assert meta["step"] == N_TOTAL // 2

    got, got_ice = _leg(_legs["model"], warm_state, warm_ice, _legs["ice_cfg"],
                        _legs["stack"], meta["step"], N_TOTAL - meta["step"])
    _assert_states_match(got, _legs["cont_state"], got_ice, _legs["cont_ice"])


# --------------------------------------------------------------- mutations
# Each case breaks ONE restored piece and asserts the comparison goes RED.
# These are the non-vacuity proof for the test above.

def _resume_with(_legs, *, state, ice, step):
    return _leg(_legs["model"], state, ice, _legs["ice_cfg"], _legs["stack"],
                step, N_TOTAL - step)


def test_dropping_the_ice_state_breaks_continuity(_legs):
    """Resume on a COLD-START pack (h=0, T_ice=260 K, S_ice=0, h_snow=0) —
    what a restart that omits the ice would rebuild."""
    got, got_ice = _resume_with(_legs, state=_legs["half_state"],
                                ice=init_dynamic_ice_state(
                                    (N_LAT, N_LON), S_ice_init=0.0),
                                step=N_TOTAL // 2)
    with pytest.raises(AssertionError):
        _assert_states_match(got, _legs["cont_state"], got_ice,
                             _legs["cont_ice"])


def test_dropping_the_step_counter_breaks_continuity(_legs):
    """Resume with step=0 — what a restart that omits the counter would do:
    the CORE-II record index restarts, so the forcing is replayed."""
    got, got_ice = _leg(_legs["model"], _legs["half_state"], _legs["half_ice"],
                        _legs["ice_cfg"], _legs["stack"], 0,
                        N_TOTAL - N_TOTAL // 2)
    with pytest.raises(AssertionError):
        _assert_states_match(got, _legs["cont_state"], got_ice,
                             _legs["cont_ice"])


def test_dropping_the_tke_carry_breaks_continuity(_legs):
    """Resume with state.tke = None — the closure re-seeds at the background
    value and the mixing (hence T/S) differs."""
    got, got_ice = _resume_with(_legs,
                                state=_legs["half_state"]._replace(tke=None),
                                ice=_legs["half_ice"], step=N_TOTAL // 2)
    with pytest.raises(AssertionError):
        _assert_states_match(got, _legs["cont_state"], got_ice,
                             _legs["cont_ice"])
