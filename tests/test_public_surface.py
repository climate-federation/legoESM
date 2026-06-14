"""Public import-surface contract for the federation carve.

The carve (FEDERATION.md) deliberately breaks two legacy surfaces in exchange
for independently-installable members — codex adversarial review flagged this, so
these tests make the break *intentional and guarded* rather than accidental:

1. ``legoesm`` is a PEP-420 namespace package, so the package-level conveniences
   the old ``legoesm/__init__.py`` exported (``legoesm.__version__``,
   ``legoesm.Field``, ``hours``/``minutes``/``days``/``years``) are GONE.  The
   version accessor is now ``importlib.metadata.version("legoesm")``.
2. The experiment-aware checkpoint/restart I/O moved OUT of the legoesm-core
   substrate (``legoesm.io``) UP to the driver layer (``legoesm.driver.*``).  A
   compatibility shim under ``legoesm.io`` is intentionally NOT provided: it
   would re-introduce an ``io -> driver`` import and break the import-linter
   independence contract — so this test asserts the move stuck.

If you are tempted to "restore" one of these surfaces, read FEDERATION.md first:
the break is the price of ``pip install legoesm-ocean`` working standalone.
"""

from __future__ import annotations

import importlib
import importlib.metadata

import pytest


def test_legoesm_is_a_namespace_package() -> None:
    import legoesm

    # PEP-420 namespace packages have no __file__ and a (possibly multi-root)
    # __path__ — this is what lets sibling members contribute subpackages.
    assert getattr(legoesm, "__file__", None) is None
    assert list(legoesm.__path__)


def test_version_accessor_is_importlib_metadata() -> None:
    # The canonical version accessor after the carve (legoesm.__version__ is gone
    # with the package initializer; legoesm._version.__version__ also works).
    version = importlib.metadata.version("legoesm")
    assert version
    from legoesm._version import __version__

    assert __version__  # single-source module-level accessor still resolves


def test_package_level_conveniences_are_removed() -> None:
    import legoesm

    # Deliberately gone with the PEP-420 conversion (were unused internally).
    for name in ("__version__", "Field", "hours", "minutes", "days", "years"):
        assert not hasattr(legoesm, name), (
            f"legoesm.{name} reappeared — the namespace carve removed the "
            f"package-level API; do not re-add a legoesm/__init__.py shim "
            f"(it would shadow sibling members' namespace portions)."
        )


def test_io_substrate_surface_is_pure() -> None:
    import legoesm.io as io

    # io keeps only substrate-level I/O after the re-scope.
    for name in (
        "CFWriter",
        "compute_state_digest",
        "pytree_state_digest",
        "save_state_checkpoint",
    ):
        assert hasattr(io, name), f"legoesm.io lost substrate export {name!r}"

    # The experiment-aware modules MUST NOT exist under legoesm.io — a shim here
    # would re-add io -> driver and break the legoesm-core independence contract.
    for moved in ("restart", "checkpoint", "distributed_checkpoint"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"legoesm.io.{moved}")


def test_experiment_io_moved_to_driver() -> None:
    # The moved APIs now live in the driver layer.
    from legoesm.driver.restart import save_restart, load_restart  # noqa: F401
    from legoesm.driver.checkpoint import load_checkpoint_auto  # noqa: F401
    from legoesm.driver.distributed_checkpoint import (  # noqa: F401
        save_checkpoint_distributed,
    )
