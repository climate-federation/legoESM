"""git_trust_repo.sh lets job-side git read a repo it would refuse as
"dubious ownership", so run manifests record the commit instead of ""."""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_HELPER = _ROOT / "scripts" / "cluster" / "levante" / "git_trust_repo.sh"
_CHAIN = _ROOT / "scripts" / "cluster" / "levante" / "amip_mpas_gpu_chain.sbatch"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "r"
    r.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_CONFIG_")}
    subprocess.run(["git", "init", "-q", str(r)], check=True, env=env)
    subprocess.run(["git", "-C", str(r), "-c", "user.email=a@b", "-c", "user.name=a",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True, env=env)
    sha = subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"], check=True, env=env,
                         capture_output=True, text=True).stdout.strip()
    for k in list(os.environ):
        if k.startswith("GIT_CONFIG_"):
            monkeypatch.delenv(k)
    # git's own switch for simulating a repo owned by someone else
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    return r, sha


def _env_after_sourcing(path, pre=""):
    out = subprocess.run(["bash", "-c", f'{pre} source "{_HELPER}" "{path}"; env -0'],
                         check=True, capture_output=True, text=True).stdout
    return dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)


def test_helper_makes_the_refused_repo_readable(repo):
    r, sha = repo
    refused = subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"],
                             capture_output=True, text=True)
    assert refused.returncode != 0 and "dubious ownership" in refused.stderr
    env = _env_after_sourcing(r)
    got = subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"], env=env,
                         capture_output=True, text=True)
    assert got.stdout.strip() == sha


def test_helper_appends_to_existing_env_config(repo):
    r, _ = repo
    env = _env_after_sourcing(r, "export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.abbrev "
                                 "GIT_CONFIG_VALUE_0=12;")
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert (env["GIT_CONFIG_KEY_0"], env["GIT_CONFIG_VALUE_0"]) == ("core.abbrev", "12")
    assert (env["GIT_CONFIG_KEY_1"], env["GIT_CONFIG_VALUE_1"]) == ("safe.directory", str(r))


def test_manifest_provenance_reads_the_commit_under_the_helper(repo, monkeypatch):
    from legoesm.io.git_provenance import git_provenance
    r, sha = repo
    (r / "anchor.py").write_text("")
    assert git_provenance(r / "anchor.py").commit == ""
    for k, v in _env_after_sourcing(r).items():
        if k.startswith("GIT_CONFIG_"):
            monkeypatch.setenv(k, v)
    assert git_provenance(r / "anchor.py").commit == sha


def test_chain_sources_the_helper_before_its_first_git_call():
    lines = _CHAIN.read_text().splitlines()
    src = next(i for i, ln in enumerate(lines) if "git_trust_repo.sh" in ln)
    first_git = next(i for i, ln in enumerate(lines) if "git -C" in ln)
    assert src < first_git
