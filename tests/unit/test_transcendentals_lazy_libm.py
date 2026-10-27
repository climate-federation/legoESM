"""``legoesm.core.transcendentals`` must import on hosts without glibc."""
from __future__ import annotations

import subprocess
import sys

import pytest
from legoesm.core import transcendentals


def test_import_does_not_load_libm():
    code = (
        "import legoesm.core.transcendentals as t, "
        "legoesm.ocean.forcing.nemo_fld_read; "
        "assert t._LIBM is None, t._LIBM"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_libm_raises_clear_error(monkeypatch):
    monkeypatch.setattr(transcendentals, "_LIBM", None)
    monkeypatch.setattr(transcendentals, "_LIBM_SONAME", "libdoes-not-exist.so.0")
    with pytest.raises(RuntimeError, match="libdoes-not-exist.so.0"):
        transcendentals.glibc_libm()
