"""Harness instrumentation of the 49r1 IFS generator random_numbers_mix.F90.

When the environment variable ECRAD49_RNG_DUMP names a file, every
INITIALIZE_RANDOM_NUMBERS call writes "SEED <kseed>" and every
UNIFORM_DISTRIBUTION call made outside initialisation writes "U <n>"
followed by its n draws (ES26.17E3), in call order. The warm-up draws made
inside INITIALIZE_RANDOM_NUMBERS are not written. The generator itself is
untouched: the two public routines are renamed *_ORIG and wrapped, so every
RETURN path of the original is covered. The JAX port's oracle test replays
this stream (user decision A16).

usage: python -I patch_rng_dump.py <random_numbers_mix.F90>   (edits in place)
"""
import re
import sys

WRAPPERS = """
SUBROUTINE INITIALIZE_RANDOM_NUMBERS (KSEED, YD_STREAM)
  INTEGER(KIND=JPIM)      , INTENT(IN   ) :: KSEED
  TYPE(RANDOMNUMBERSTREAM), INTENT(INOUT) :: YD_STREAM
  CALL RNG_DUMP_OPEN()
  IF (LLDUMP) THEN
    WRITE(NUDUMP,'(A,1X,I0)') 'SEED', KSEED
    FLUSH(NUDUMP)
  ENDIF
  LLINIT_DUMP = .TRUE.
  CALL INITIALIZE_RANDOM_NUMBERS_ORIG (KSEED, YD_STREAM)
  LLINIT_DUMP = .FALSE.
END SUBROUTINE INITIALIZE_RANDOM_NUMBERS

SUBROUTINE UNIFORM_DISTRIBUTION (PX,YD_STREAM)
  TYPE(RANDOMNUMBERSTREAM)     , INTENT(INOUT) :: YD_STREAM
  REAL(KIND=JPRB), DIMENSION(:), INTENT(  OUT) :: PX
  CALL UNIFORM_DISTRIBUTION_ORIG (PX, YD_STREAM)
  CALL RNG_DUMP_OPEN()
  IF (LLDUMP .AND. .NOT. LLINIT_DUMP) THEN
    WRITE(NUDUMP,'(A,1X,I0)') 'U', SIZE(PX)
    IF (SIZE(PX) > 0) WRITE(NUDUMP,'(ES26.17E3)') PX
    FLUSH(NUDUMP)
  ENDIF
END SUBROUTINE UNIFORM_DISTRIBUTION

SUBROUTINE RNG_DUMP_OPEN()
  CHARACTER(LEN=1024) :: CLF
  ! NEWUNIT returns a negative unit number, so "open" is tracked by LLDUMP
  IF (LLOPENED_DUMP) RETURN
  LLOPENED_DUMP = .TRUE.
  CALL GET_ENVIRONMENT_VARIABLE('ECRAD49_RNG_DUMP', CLF)
  LLDUMP = LEN_TRIM(CLF) > 0
  IF (LLDUMP) OPEN(NEWUNIT=NUDUMP, FILE=TRIM(CLF), STATUS='REPLACE', ACTION='WRITE')
END SUBROUTINE RNG_DUMP_OPEN
"""


def sub(s, pattern, repl):
    s2, n = re.subn(pattern, repl, s, flags=re.M)
    if n != 1:
        raise SystemExit(f"pattern matched {n} times: {pattern!r}")
    return s2


def main(path):
    s = open(path).read()
    for name in ("INITIALIZE_RANDOM_NUMBERS", "UNIFORM_DISTRIBUTION"):
        s = sub(s, rf"^SUBROUTINE {name} *\(", f"SUBROUTINE {name}_ORIG (")
        s = sub(s, rf"^END SUBROUTINE {name}\b", f"END SUBROUTINE {name}_ORIG")
    # module-scope dump state, right before CONTAINS
    s = sub(s, r"^CONTAINS\s*$",
            "INTEGER, PRIVATE :: NUDUMP = 0\nLOGICAL, PRIVATE :: LLDUMP = .FALSE., LLOPENED_DUMP = .FALSE., LLINIT_DUMP = .FALSE.\nCONTAINS")
    s = sub(s, r"^END MODULE RANDOM_NUMBERS_MIX", WRAPPERS + "\nEND MODULE RANDOM_NUMBERS_MIX")
    open(path, "w").write(s)


if __name__ == "__main__":
    main(sys.argv[1])
