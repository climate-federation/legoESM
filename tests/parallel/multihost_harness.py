"""Shared launcher for the 2-process jax.distributed (multihost) gates.

Binding a probe socket to port 0 and releasing it leaves a window in which
another process can claim the port before the jax coordinator binds it — a
real flake under parallel pytest on a busy runner (codex round-4). The
launcher therefore RETRIES the whole federation on a fresh port when the
failure output looks like a coordinator bind/connect problem, and only
surfaces genuine (numerics/assertion) failures to the caller.
"""
from __future__ import annotations

import socket
import subprocess

# Coordinator-level failure signatures (retry-worthy); anything else is a
# real failure the caller must see.
_BIND_FAILURE_MARKERS = (
    "address already in use",
    "failed to bind",
    "bind address",
    "unavailable: connection",
    "deadline exceeded",
    "failed to connect to coordinat",
)

N_TRIES = 3


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _looks_like_port_race(outputs: list[str]) -> bool:
    blob = "\n".join(o or "" for o in outputs).lower()
    return any(m in blob for m in _BIND_FAILURE_MARKERS)


def run_federated(build_cmd, n_proc: int, env: dict, timeout_s: int):
    """Run ``n_proc`` federated workers; retry on coordinator port races.

    ``build_cmd(rank, port) -> list[str]`` builds each worker's argv.
    Returns ``(returncodes, outputs)`` of the LAST attempt. Raises
    ``TimeoutError`` if every attempt hangs.
    """
    last = None
    for attempt in range(N_TRIES):
        port = free_port()
        procs = [
            subprocess.Popen(
                build_cmd(rank, port), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for rank in range(n_proc)
        ]
        outs: list[str] = []
        timed_out = False
        for p in procs:
            try:
                out, _ = p.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                out = ""
            outs.append(out)
        if timed_out:
            for p in procs:
                p.kill()
            for p in procs:
                p.wait()
            last = ([p.returncode for p in procs], outs)
            continue  # a hang can also be a lost coordinator race — retry
        rcs = [p.returncode for p in procs]
        last = (rcs, outs)
        if all(rc == 0 for rc in rcs):
            return last
        if not _looks_like_port_race(outs):
            return last  # genuine failure — surface it
        # else: port race — loop with a fresh port
    if last is None or (last and not last[1]):
        raise TimeoutError(
            f"federated workers failed on all {N_TRIES} attempts")
    return last
