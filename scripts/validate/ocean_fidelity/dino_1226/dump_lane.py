"""#1455: which NEMO dump run (and therefore which ocean STATE) a #1226 probe
measures against.  ONE shared selector -- never a per-probe copy.

The #1226 term sweep was measured almost entirely against ``RUN_GDB``, a
1-rank NEMO run from the YEAR-5 restart whose ``kt==nit000`` dumps are the
per-term reference.  The 90-day twin, however, starts at DAY 180, and its ACC
sits +1.87 Sv above NEMO's -- flat across the window, against a 0.091 Sv noise
floor.  A term that is at the bar on the year-5 state is NOT thereby at the
bar on the day-180 state: the operands differ.  Re-measuring the sweep at the
twin's own starting point is the point of this module.

``RUN_SEQDUMP_D180_1R`` is a drop-in substitute for ``RUN_GDB``: same NEMO
config, same 1-rank decomposition (jpni=jpnj=1, nn_hls=2), so every dump has
the identical global haloed shape (3-D 56x203x35 f8, 2-D 56x203) and the same
``kt==nit000`` semantics -- only the restart it starts from differs.  Its
restart is symlinked into the run dir under its own name so
``os.path.join(RUN_DIR, RESTART)`` needs no special case, and the SEQ-DUMP
families that carry a ``_ktNNNNNNNN`` suffix are symlinked to their
un-suffixed names AT THE FIRST STEP ONLY (kt=5761=nit000), which is exactly
what RUN_GDB's un-suffixed files are.

NOT available on the d180 lane: the ``ldf_in_*``/``ldf_out_*``/``r3c_in_*``/
``r3c_out_*``/``sh2_in_*``/``sh2_out_*``/``zad_in_*``/``zad_out_*`` unit-harness
families.  Those come from a later oracle instrumentation that the preserved
SEQ-DUMP binary (md5 ea0c113c) does not contain, so probes reading them can
only be re-measured after that instrumentation is rebuilt.  ``dump_path``
raises naming the file rather than letting a probe fall through to something
else.

Select with ``DINO_1226_LANE=gdb_y5`` (default, the sweep's historical lane)
or ``DINO_1226_LANE=d180``.  An unknown value raises (dispatch hardening) --
a typo must never silently select the historical lane and produce a number
that gets recorded as a day-180 measurement.
"""
from __future__ import annotations

import os

_DINO_CFG = os.environ.get(
    "DINO_ORACLE_ROOT",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")

# lane -> (run dir, restart basename INSIDE that dir, kt of the dumps)
LANES = {
    # The #1226 sweep's historical lane: NEMO year 5, 1 rank, dumps at nit000.
    "gdb_y5": (os.path.join(_DINO_CFG, "RUN_GDB"),
               "DINO_00057600_restart.nc", 57601),
    # The 90-day twin's OWN starting point: day 180 (adatrj=180.0), 1 rank,
    # instrumented binary md5 ea0c113c, dumps at nit000=5761.  Controlled in
    # the oracle repo's .binary_provenance.txt: its step-5764 restart is
    # bit-identical to the certified binary's from the same restart.
    "d180": (os.path.join(_DINO_CFG, "RUN_SEQDUMP_D180_1R"),
             "DINO_00005760_restart.nc", 5761),
    # NEMO year 20, 1 rank, SAME instrumented binary and SAME symlink
    # convention as d180.  Present because several sweep rows were originally
    # measured here rather than on RUN_GDB, and because RUN_GDB's ocean.output
    # lacks strings some probes parse -- so this is the lane that can give a
    # year-20 BASELINE for a probe RUN_GDB cannot run at all.
    "y20": (os.path.join(_DINO_CFG, "RUN_SEQDUMP_Y20_1R"),
            "DINO_00230400_restart.nc", 230401),
}

LANE = os.environ.get("DINO_1226_LANE", "gdb_y5")
if LANE not in LANES:
    raise SystemExit(
        f"Unknown DINO_1226_LANE={LANE!r}: expected one of "
        f"{sorted(LANES)}. Refusing to guess -- a typo here would record a "
        "year-5 number as a day-180 measurement.")

RUN_DIR, RESTART, KT_DUMP = LANES[LANE]
#: seconds of the seasonal clock at the step the dumps describe.  NEMO
#: evaluates the surface forcing of step kt at kt*rn_Dt (nn_fsbc=1), so this
#: is the value a probe must pass as ``t_seconds``; deriving it from KT_DUMP
#: is what keeps the two lanes in their own seasons instead of both at day 0.
DT = 2700.0
T_SECONDS = KT_DUMP * DT


def banner() -> str:
    return (f"DUMP LANE: {LANE}  run_dir={RUN_DIR}  restart={RESTART}  "
            f"kt(nit000)={KT_DUMP}  t_seconds={T_SECONDS:.0f}s "
            f"(= day {T_SECONDS/86400.0:.2f})")


def dump_path(basename: str) -> str:
    """Absolute path of a dump, or raise naming what was looked for.

    Fail-closed on purpose: a probe that silently skips a missing dump would
    report the lane as measured when it is not.
    """
    p = os.path.join(RUN_DIR, basename)
    if not os.path.exists(p):
        raise SystemExit(
            f"dump {basename!r} does not exist on lane {LANE!r} ({RUN_DIR}). "
            "If this is one of the unit-harness families (ldf_in/ldf_out/"
            "r3c_in/r3c_out/sh2_in/sh2_out/zad_in/zad_out), the preserved "
            "SEQ-DUMP binary does not write it and the row cannot be "
            "re-measured on this lane without rebuilding that instrumentation.")
    return p


def restart_path() -> str:
    return dump_path(RESTART)
