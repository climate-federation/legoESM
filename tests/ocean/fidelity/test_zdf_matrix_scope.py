"""The zdf-matrix instrument's scope check must be able to FAIL.

The check exists because ``run.sh`` once inserted ``IF( kt <= nit000 + 1 )``
into ``tra_zdf_imp``, which has no ``kt``, and the build died a minute later.
A check that only ever passes would have been worthless, so this test puts
``kt`` back and requires the refusal, and it also requires the corrected
instrument to pass -- both directions, on the real compiled oracle source.

Skipped when the oracle tree is not on this machine.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_NEMO = pathlib.Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
_PPSRC = _NEMO / "cfgs/DINO/BLD/ppsrc/nemo/trazdf.f90"
_HERE = pathlib.Path(__file__).resolve().parents[3]
_CHECK = (_HERE / "scripts/validate/ocean_fidelity/dino_1226"
          / "nemo_dino_zdf_matrix" / "check_scope.py")

pytestmark = pytest.mark.skipif(
    not _PPSRC.exists(), reason="NEMO DINO compiled source not present here")


def _load():
    spec = importlib.util.spec_from_file_location("_check_scope", _CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The dump block as run.sh inserts it, reduced to the statements that carry
#: symbols.  Kept literal so the test does not re-run the patcher.
_GOOD = """
            ! ---- #1728 WRITE-ONLY matrix dump (kt = nit000, nit000+1) ----
            IF( nzdfmat_kt >= 0 .AND. nzdfmat_kt <= nit000 + 1 ) THEN
               WRITE( cl_zm, '("trazdf_",A3,"_kt",I8.8,"_rank",I2.2,".bin")' )  &
                  &  'zwi', nzdfmat_kt, narea - 1
               OPEN( UNIT=9201, FILE=TRIM(cl_zm), ACCESS='STREAM',              &
                  &  FORM='UNFORMATTED', STATUS='REPLACE' )
               DO jk2 = 1, jpk
                  WRITE(9201) ( zwi(ji2,jk2), ji2 = 1, jpi )
               END DO
               CLOSE(9201)
            ENDIF
"""
#: The SAME block with the defect that failed the build restored.
_BAD = _GOOD.replace("nzdfmat_kt >= 0 .AND. nzdfmat_kt <= nit000 + 1",
                     "kt <= nit000 + 1")


def _run(block):
    mod = _load()
    return mod.check(_PPSRC, mod.ANCHOR, block, mod.BLOCK_RE,
                     mod.DECLARED_BY_PATCH)


def test_corrected_instrument_is_in_scope():
    assert _run(_GOOD) == "tra_zdf_imp"


def test_the_defect_that_broke_the_build_is_refused():
    with pytest.raises(SystemExit) as exc:
        _run(_BAD)
    msg = str(exc.value)
    assert "'kt'" in msg or "['kt'" in msg, msg
    assert "tra_zdf_imp" in msg


def test_module_scope_names_are_not_vacuous():
    """The allowed set must really come from the compiled tree."""
    mod = _load()
    names = mod.module_scope_names(_PPSRC.parent)
    # things that ARE module variables in NEMO
    assert {"jpi", "jpk", "narea", "nit000"} <= names
    # `kt` is always a dummy argument, never a module variable -- that is the
    # whole discriminator, so assert it explicitly.
    assert "kt" not in names
