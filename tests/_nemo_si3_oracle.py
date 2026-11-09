"""Skip marker for SI3 NEMO-oracle tests whose records are not mounted.

The SI3 phase-2 gate tests read oracle records under
``/data/abyssal/dbalwada/nemo-testcases-l3/...``, which exist only on the
machine that produced them. Elsewhere those tests can only raise
``FileNotFoundError``, which says nothing about the model. ``needs_oracle``
skips them when the record directory is absent. Setting
``LEGOESM_REQUIRE_NEMO_ORACLE=1`` disables the skip, so a machine that is
supposed to hold the records fails loudly instead of skipping silently.
A directory that exists but is incomplete is NOT skipped: the test runs and
fails on the missing file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

STRICT_ENV = "LEGOESM_REQUIRE_NEMO_ORACLE"


def needs_oracle(root: Path) -> pytest.MarkDecorator:
    absent = not Path(root).is_dir() and os.environ.get(STRICT_ENV) != "1"
    return pytest.mark.skipif(
        absent,
        reason=f"NEMO SI3 oracle records absent: {root} (set {STRICT_ENV}=1 to require)",
    )
