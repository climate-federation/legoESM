"""Convert CloudBench LES cases into LES-suite artifacts (PI 2026-10-31).

Each case is streamed from the public bucket and only its mean profiles and fluxes
are written (``<out>/<case>.npz``); see ``legoesm.atmosphere.les_suite.cloudbench``.
A finished case is skipped only if its file loads and was made by the same converter
code (hash of every legoesm source file the converter imports); otherwise it is
rebuilt, so the job can be resubmitted until all are done. A case that fails is
logged to ``<out>/failures.jsonl`` and the rest continue. Files are written under
``<out>/.partial/`` and moved into place when complete. If a worker process dies
(e.g. out of memory) the batch stops, logs the unfinished cases and exits 1; a
resubmit picks them up (PI 2026-11-01). The held-out
site list (fixed seed, PI 2026-10-31) is written once to ``<out>/held_out_sites.json``.

usage: build_cloudbench_artifacts.py --forcing-dir DIR --out-dir DIR
       [--exp amip] [--shard I N] [--workers K]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path


def list_cases(forcing_dir: Path, exp: str) -> list[tuple[int, int]]:
    """Every (site, month) with a forcing directory for ``exp``, sorted."""
    out = []
    for site in forcing_dir.iterdir():
        if not (site.is_dir() and site.name.isdigit()):
            continue
        for month in site.iterdir():
            if month.is_dir() and month.name.isdigit() and (month / exp).is_dir():
                out.append((int(site.name), int(month.name)))
    return sorted(out)


def shard(cases, index: int, count: int):
    if not 0 <= index < count:
        raise SystemExit(f"--shard {index} {count}: need 0 <= I < N")
    return cases[index::count]


def _convert(job):
    site, month, exp, forcing_dir, out_dir, prov = job
    from legoesm.atmosphere.les_suite import cloudbench as cb  # noqa: PLC0415
    from legoesm.atmosphere.les_suite.bridge import save_artifact  # noqa: PLC0415

    name = cb.case_name(site, month, exp)
    final = Path(out_dir) / f"{name}.npz"
    if _is_current(final, prov["code_sha256"]):
        return name, "skipped", 0.0, None
    t0 = time.time()
    tmp = final.parent / ".partial" / f"{final.stem}.{socket.gethostname()}-{os.getpid()}.npz"
    try:
        tmp.parent.mkdir(exist_ok=True)
        art = cb.build_artifact(site, month, exp, forcing_dir)
        art.les_setup["converter"] = prov
        save_artifact(art, tmp)
        os.replace(tmp, final)
        return name, "done", time.time() - t0, None
    except Exception:  # noqa: BLE001 - logged per case, the batch continues
        tmp.unlink(missing_ok=True)
        return name, "failed", time.time() - t0, traceback.format_exc()


def _is_current(path: Path, code_sha256: str) -> bool:
    """True if ``path`` loads as an artifact made by converter code ``code_sha256``."""
    from legoesm.atmosphere.les_suite.bridge import load_artifact  # noqa: PLC0415
    if not path.exists():
        return False
    try:
        return load_artifact(path).les_setup["converter"]["code_sha256"] == code_sha256
    except Exception:  # noqa: BLE001 - any unreadable or foreign file is rebuilt
        return False


def _converter_code_sha256() -> str:
    """Hash of every imported legoesm source file (call after importing the converter)."""
    import hashlib  # noqa: PLC0415
    h = hashlib.sha256()
    for name in sorted(sys.modules):
        f = getattr(sys.modules[name], "__file__", None)
        if (name == "legoesm" or name.startswith("legoesm.")) and f and f.endswith(".py"):
            h.update(name.encode() + b"\0" + Path(f).read_bytes() + b"\0")
    return h.hexdigest()


def _git(*cmd) -> str:
    try:
        return subprocess.run(["git", *cmd], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _code_provenance() -> dict:
    """HEAD plus a hash of the uncommitted diff, so two dirty trees differ."""
    import hashlib  # noqa: PLC0415
    diff = _git("diff", "HEAD")
    return {"git_sha": _git("rev-parse", "HEAD").strip(), "dirty": bool(diff.strip()),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def _write_held_out(path: Path, record: dict) -> None:
    """Create ``path`` with ``record`` exactly once (atomic, exclusive); if it exists,
    refuse unless it holds the same record."""
    tmp = path.with_name(f"{path.name}.tmp-{socket.gethostname()}-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(record, indent=1))
        os.link(tmp, path)
    except FileExistsError:
        pass
    finally:
        tmp.unlink(missing_ok=True)
    if json.loads(path.read_text()) != record:
        raise SystemExit(f"{path} differs from this draw; refusing to mix splits")


def _append_line(path: Path, obj: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    data = (json.dumps(obj) + "\n").encode()
    try:
        if os.write(fd, data) != len(data):
            raise OSError(f"{path}: short write of a log record")
    finally:
        os.close(fd)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--forcing-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--exp", default="amip")
    p.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("I", "N"))
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    from legoesm.atmosphere.les_suite import cloudbench as cb  # noqa: PLC0415

    cases = list_cases(args.forcing_dir, args.exp)
    if not cases:
        raise SystemExit(f"{args.forcing_dir}: no {args.exp} cases")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sites = sorted({s for s, _ in cases})
    record = {"held_out_sites": cb.held_out_sites(sites), "fraction": cb.HELD_OUT_FRACTION,
              "seed": cb.HELD_OUT_SEED, "n_sites": len(sites)}
    _write_held_out(args.out_dir / "held_out_sites.json", record)
    mine = shard(cases, *args.shard)
    prov = {**_code_provenance(), "code_sha256": _converter_code_sha256(), "argv": sys.argv[1:] if argv is None else list(argv),
            "host": socket.gethostname(), "pbs_jobid": os.environ.get("PBS_JOBID")}
    jobs = [(s, m, args.exp, str(args.forcing_dir), str(args.out_dir), prov)
            for s, m in mine]
    print(f"[cloudbench] {json.dumps({**prov, 'n_cases_total': len(cases), 'n_cases_shard': len(jobs)})}",
          flush=True)
    counts = {"done": 0, "skipped": 0, "failed": 0}
    def record(fut):
        name, status, secs, tb = fut.result()
        counts[status] += 1
        print(f"[cloudbench] {name} {status} {secs:.0f}s", flush=True)
        if tb is not None:
            _append_line(args.out_dir / "failures.jsonl",
                         {"case": name, "converter": prov, "error": tb})

    futures, seen = {}, set()
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        try:
            for j in jobs:
                futures[pool.submit(_convert, j)] = j
            for fut in as_completed(futures):
                record(fut)
                seen.add(fut)
        except BrokenProcessPool:
            left = []
            for f, j in futures.items():
                if f not in seen and f.done() and f.exception() is None:
                    record(f)
                elif f not in seen:
                    left.append(cb.case_name(j[0], j[1], j[2]))
            left += [cb.case_name(*j[:3]) for j in jobs[len(futures):]]
            left.sort()
            _append_line(args.out_dir / "failures.jsonl",
                         {"case": None, "converter": prov, "unfinished": left,
                          "error": "a worker process died; batch stopped"})
            print(f"[cloudbench] worker died; stopped with {len(left)} unfinished",
                  flush=True)
            counts["failed"] += len(left)
    print(f"[cloudbench] finished {json.dumps(counts)}", flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
