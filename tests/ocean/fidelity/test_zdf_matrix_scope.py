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
            ! ---- #1728 END matrix dump ----
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


#: The block with a NESTED IF/ENDIF and a bogus symbol AFTER it.  The first
#: version of BLOCK_RE stopped at the first ENDIF and never saw the symbol.
_NESTED = _GOOD.replace("""               CLOSE(9201)""",
"""               IF( narea == 1 ) THEN
                  WRITE(9201) zwi(1,1)
               ENDIF
               CLOSE(9201)
               IF( zz_bogus_1728 > 0 ) WRITE(9201) zwi(1,1)""")


def test_a_symbol_after_a_nested_endif_is_still_checked():
    with pytest.raises(SystemExit) as exc:
        _run(_NESTED)
    assert "zz_bogus_1728" in str(exc.value)


def test_module_scope_names_are_not_vacuous():
    """The allowed set must really come from the compiled tree."""
    mod = _load()
    names = mod.module_scope_names(_PPSRC.parent)
    # things that ARE module variables in NEMO
    assert {"jpi", "jpk", "narea", "nit000"} <= names
    # `kt` is always a dummy argument, never a module variable -- that is the
    # whole discriminator, so assert it explicitly.
    assert "kt" not in names


def test_the_allowed_set_is_this_file_s_use_closure_not_the_whole_tree():
    """A module variable this file cannot reach must NOT be accepted.

    The first version harvested every module-scope declaration in the compiled
    tree (3340 names), so ``x`` -- a module variable in ``storng.f90``, which
    ``trazdf`` does not USE at any depth -- passed a check gfortran rejects.
    An independent review found it.
    """
    mod = _load()
    src = _PPSRC.read_text(errors="replace")
    closure = mod.module_scope_names(_PPSRC.parent, seed_text=src)
    whole = mod.module_scope_names(_PPSRC.parent)
    assert len(closure) < len(whole), (len(closure), len(whole))
    # still reachable, at one and two USE levels
    assert {"jpi", "jpk", "narea", "nit000"} <= closure
    assert "kt" not in closure
    bogus = _GOOD.replace("nzdfmat_kt >= 0", "x >= 0")
    with pytest.raises(SystemExit) as exc:
        _run(bogus)
    assert "'x'" in str(exc.value), str(exc.value)
