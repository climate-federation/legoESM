"""Helper for translating ``ModelDriver.run()`` status strings to
process exit codes.

iter-109 (codex iter-104 MEDIUM-8): centralizes the
status-string-to-exit-code mapping so all user-facing CLI
wrappers (``src/legoesm/cli.py``,
``scripts/run_held_suarez_rrtmgp_allgrids.py``,
``scripts/run_held_suarez_icos_0p5deg.py``, etc.) can share a
single contract.

The driver returns a string status (see ``ModelDriver.run``):

* ``"COMPLETED"``        — clean run, exit code 0.
* ``"BLOWUP at day N"``  — numerical instability detected,
                            exit code 1.
* anything else          — unexpected; exit code 1 (defensive).

iter-97 / iter-100 / iter-101 unified the exit-code contract
for the four primary user-facing entry points
(``run_omip.py``, ``run_amip.py``, ``run_rce.py``, the matrix
runners).  iter-109 closes the same gap for the OTHER
ModelDriver wrappers that codex iter-104 review identified.
"""
from __future__ import annotations


def status_to_exit_code(status: str) -> int:
    """Translate a ``ModelDriver.run()`` status string to an
    exit code (0 = success, 1 = failure).

    Parameters
    ----------
    status
        Run status string from ``ModelDriver.run()``.  Conventionally
        ``"COMPLETED"`` for success or ``"BLOWUP at day N"`` for
        numerical instability.  Any other value is treated as
        failure (exit 1) so unexpected statuses don't slip through.

    Returns
    -------
    int
        ``0`` if ``status == "COMPLETED"``, ``1`` otherwise.
    """
    if status == "COMPLETED":
        return 0
    return 1
