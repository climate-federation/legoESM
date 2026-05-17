"""Veros ACC channel adapter.

Wraps Veros's built-in :class:`veros.setups.acc.acc.ACCSetup` (a 30x42x15
re-entrant zonal channel adapted from pyOM2). The default ``runlen`` is
shortened so the fidelity layer can integrate a short reference run
(typically 1-10 days) for comparison against legoESM's ``acc_channel``
experiment, while the legoESM-side equilibrium DINO/ACC-style runs stay
on the legoESM model.
"""

from __future__ import annotations

DEFAULT_RUNLEN_S: float = 30.0 * 86400.0  # 30 days for a quick comparison run


def make_setup(*, runlen_s: float = DEFAULT_RUNLEN_S):
    """Return a fresh ACCSetup with a shortened ``runlen``.

    The Veros object is imported lazily so this module can be imported even
    when Veros is not installed.
    """
    from veros.setups.acc.acc import ACCSetup

    setup = ACCSetup()
    # Note: settings.runlen is mutated post-`setup()` by the runner, so we
    # only stash the desired value here for transparency.
    setup._legoesm_target_runlen_s = float(runlen_s)
    return setup
